"""Archive streamed LLM reasoning and answers, including interrupted responses."""
import json
import time
import uuid
import queue
import threading

from experiment_runtime import RequestArchive, atomic_json, redact, utc
from kimi_request_parameters import normalize_kimi_parameters
from model_context import input_metrics
from backend_hooks import request_audit as RequestAudit


class RequestDeadlineExceeded(TimeoutError):
    pass


class NoOutputTimeout(TimeoutError):
    pass


class StreamingRequestFailure(RuntimeError):
    """Public failure metadata without discarding the original exception type."""
    def __init__(self, message, *, failure_kind, error_type, phase, archive_ref,
                 output_received=None, chunks_received=None):
        super().__init__(message)
        self.failure_kind=failure_kind
        self.error_type=error_type
        self.phase=phase
        self.archive_ref=str(archive_ref)
        self.output_received=output_received
        self.chunks_received=chunks_received


def request_timeout_details(error):
    """Normalize deadline wrappers without treating arbitrary stream errors as timeouts."""
    chain=[]
    current=error
    while current is not None and all(current is not item for item in chain) and len(chain)<8:
        chain.append(current)
        current=current.__cause__ or current.__context__
    wall_kinds={'request_wall_time_exceeded','stream_wall_time_exceeded'}
    kind=next((getattr(item,'failure_kind',None) for item in chain
               if getattr(item,'failure_kind',None)),None)
    timeout=kind in wall_kinds|{'no_output_timeout','transport_timeout'} or any(
        isinstance(item,TimeoutError) or type(item).__name__ in
        {'APITimeoutError','TimeoutExpired','RequestDeadlineExceeded','NoOutputTimeout'} for item in chain)
    if not timeout:
        return None
    if kind is None:
        if any(isinstance(item,NoOutputTimeout) for item in chain):kind='no_output_timeout'
        elif any(isinstance(item,RequestDeadlineExceeded) for item in chain):kind='request_wall_time_exceeded'
        else:kind='transport_timeout'
    output=next((getattr(item,'output_received') for item in chain
                 if getattr(item,'output_received',None) is not None),None)
    category=('model_decision_timeout' if kind in wall_kinds and output is True else
              'no_output_timeout' if kind=='no_output_timeout' else
              'request_deadline' if kind in wall_kinds else 'transport_timeout')
    result=dict(failure_kind=kind,timeout_category=category,output_received=output)
    for name in ('archive_ref','error_type','phase','chunks_received'):
        value=next((getattr(item,name) for item in chain if getattr(item,name,None) is not None),None)
        if value is not None:result[name]=value
    return result


def request_protocol_details(error):
    chain=[]
    current=error
    while current is not None and all(current is not item for item in chain) and len(chain)<8:
        chain.append(current)
        current=current.__cause__ or current.__context__
    kind=next((getattr(item,'failure_kind',None) for item in chain
               if getattr(item,'failure_kind',None)=='multiple_tool_calls_when_disabled'),None)
    if kind is None:
        return None
    result=dict(failure_kind=kind)
    for name in ('archive_ref','error_type','phase','chunks_received','output_received'):
        value=next((getattr(item,name) for item in chain if getattr(item,name,None) is not None),None)
        if value is not None:result[name]=value
    return result


def _chunk_has_output(chunk):
    for choice in getattr(chunk, 'choices', None) or ():
        delta = getattr(choice, 'delta', None)
        if delta is None:
            continue
        if getattr(delta, 'reasoning_content', None) or getattr(delta, 'content', None) or getattr(delta, 'tool_calls', None):
            return True
    return False


def deadline_events(create_stream, seconds, *, no_output_seconds=None, has_output=None):
    """Bound header waits and idle reads together. Only caller writes archives.

    Network I/O runs in a daemon; on timeout close the response asynchronously
    so a blocked socket close cannot hold up the caller. Transport timeout also
    bounds an outstanding header request whose response is not available yet.
    """
    events = queue.Queue(maxsize=8)
    stopped = threading.Event()
    opened = []
    deadline = time.monotonic()+seconds
    no_output_deadline = None if no_output_seconds is None else time.monotonic()+min(seconds, no_output_seconds)
    output_seen = False

    def emit(kind, value=None):
        while not stopped.is_set():
            try:
                events.put((kind,value),timeout=.05)
                return
            except queue.Full:
                continue

    def read():
        try:
            with create_stream() as stream:
                opened.append(stream)
                if stopped.is_set():
                    return
                emit("headers")
                for chunk in stream:
                    if stopped.is_set():
                        return
                    emit("chunk",chunk)
                emit("end")
        except Exception as error:
            emit("error",error)

    def close():
        if opened:
            try:
                opened[0].close()
            except Exception:
                pass

    threading.Thread(target=read,daemon=True).start()
    try:
        while True:
            now = time.monotonic()
            remaining = deadline-now
            if remaining<=0:
                raise RequestDeadlineExceeded("Request wall-clock deadline exceeded")
            wait = remaining
            if not output_seen and no_output_deadline is not None:
                no_output_remaining = no_output_deadline-now
                if no_output_remaining<=0:
                    raise NoOutputTimeout("No model output received before no-output deadline")
                wait = min(wait, no_output_remaining)
            try:
                kind,value = events.get(timeout=wait)
            except queue.Empty:
                if not output_seen and no_output_deadline is not None and time.monotonic()>=no_output_deadline:
                    raise NoOutputTimeout("No model output received before no-output deadline") from None
                raise RequestDeadlineExceeded("Request wall-clock deadline exceeded") from None
            if time.monotonic()>=deadline:
                raise RequestDeadlineExceeded("Request wall-clock deadline exceeded")
            if kind=="error":
                raise value
            if kind=="end":
                return
            if kind=="chunk" and not output_seen:
                if (has_output(value) if has_output is not None else _chunk_has_output(value)):
                    output_seen = True
                elif no_output_deadline is not None and time.monotonic()>=no_output_deadline:
                    raise NoOutputTimeout("No model output received before no-output deadline")
            yield kind,value
    finally:
        stopped.set()
        if opened:
            threading.Thread(target=close,daemon=True).start()



class StreamingRequestArchive(RequestArchive):
    @staticmethod
    def _is_retryable_zero_output(error):
        return (isinstance(error, (NoOutputTimeout, TimeoutError, ConnectionError))
                or type(error).__name__ in {
                    'APIConnectionError', 'APITimeoutError', 'RequestDeadlineExceeded',
                    'NoOutputTimeout', 'TimeoutError',
                })

    def complete(self, label, messages, *, total_seconds=300, no_output_seconds=60,
                 allow_tool_calls=False, retries=0, retry_backoff_s=(1.0, 2.0), **parameters):
        if type(total_seconds) not in (int,float) or not 0 < total_seconds < float('inf'):
            raise ValueError('total_seconds must be positive and finite')
        profile_timeout = self.profile.get('timeout')
        if profile_timeout is not None:
            if type(profile_timeout) not in (int,float) or not 0 < profile_timeout < float('inf'):
                raise ValueError('profile timeout must be positive and finite')
            total_seconds = min(float(total_seconds), float(profile_timeout))
        if no_output_seconds is not None:
            no_output_seconds = self.profile.get('no_output_timeout', no_output_seconds)
            if type(no_output_seconds) not in (int,float) or not 0 < no_output_seconds < float('inf'):
                raise ValueError('no_output_seconds must be positive and finite')
            no_output_seconds = min(float(total_seconds), float(no_output_seconds))
        key = self.profile['api_key']
        if key and key in json.dumps(messages, ensure_ascii=False):
            raise ValueError('Credential found in outgoing message')
        requested_parameters = parameters
        parameters = normalize_kimi_parameters(self.profile['model'], parameters)
        adjustments = [field for field in sorted(requested_parameters.keys() | parameters.keys())
            if requested_parameters.get(field) != parameters.get(field)]
        request_root = self.folder / (label + '_' + uuid.uuid4().hex)
        payload = dict(model=self.profile['model'], messages=messages, **parameters,
                       stream=True, stream_options={'include_usage': True})
        metrics = input_metrics(messages, parameters.get('tools', ()))
        overall_deadline = time.monotonic() + float(total_seconds)
        max_attempts = 1 + max(0, int(retries))
        for attempt_number in range(1, max_attempts + 1):
            attempt = request_root / f'attempt_{attempt_number:03d}'
            attempt.mkdir(parents=True, exist_ok=(attempt_number == 1))
            atomic_json(attempt / 'request.json', redact(dict(
                started_at=utc(), endpoint=self.profile['base_url'], payload=payload,
                parameter_adjustments=adjustments, attempt_number=attempt_number), [key]))
            atomic_json(attempt / 'input-metrics.json', metrics)
            audit = RequestAudit(dict(payload, timeout=total_seconds), endpoint=self.profile['base_url'], label=label,
                                 secrets=[key], archive_ref=str(attempt), logical_key=str(self.folder.resolve())+'/'+label)
            started = time.monotonic()
            attempt_budget = max(0., overall_deadline - started)
            attempt_no_output = None if no_output_seconds is None else min(attempt_budget, float(no_output_seconds))
            reasoning = content = ''
            usage = None
            finish = None
            returned_model = None
            response_id = None
            timings = {}
            tool_calls = {}
            phase = 'waiting_response_headers'
            chunks_received = 0
            output_received = False
            failure_kind = None
            tool_delta_count = 0
            tool_delta_sample = []

            def save():
                # Rewrite accumulated text with redaction so split chunks remain safe.
                atomic_json(attempt / 'partial-response.json', redact(dict(
                    reasoning_content=reasoning, content=content, usage=usage,
                    tool_calls=list(tool_calls.values()),
                    tool_stream_diagnostics=dict(delta_count=tool_delta_count, sample=tool_delta_sample, sample_limit=64),
                    finish_reason=finish, timing=timings,
                    elapsed_seconds=round(time.monotonic() - started, 3)), [key]))

            try:
                for event, chunk in deadline_events(
                        lambda: self.client.chat.completions.create(**payload, timeout=max(0.1, attempt_budget)),
                        max(0., overall_deadline - time.monotonic()),
                        no_output_seconds=attempt_no_output):
                    if event == 'headers':
                        timings['response_headers_seconds'] = round(time.monotonic()-started,3)
                        phase = 'reading_response_stream'
                        continue
                    chunks_received += 1
                    audit.first_response()
                    elapsed = round(time.monotonic() - started, 3)
                    timings.setdefault('first_chunk_seconds', elapsed)
                    returned_model = chunk.model
                    response_id = chunk.id
                    if chunk.usage is not None:
                        usage = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        delta = choice.delta
                        thought = getattr(delta, 'reasoning_content', None) or ''
                        answer = delta.content or ''
                        if thought:
                            output_received = True
                            timings.setdefault('first_reasoning_seconds', elapsed)
                            timings['last_reasoning_seconds'] = elapsed
                        if answer:
                            output_received = True
                            timings.setdefault('first_answer_seconds', elapsed)
                            timings['last_answer_seconds'] = elapsed
                        reasoning += thought
                        content += answer
                        if delta.tool_calls:
                            output_received = True
                            timings.setdefault('first_tool_call_seconds', elapsed)
                            timings['last_tool_call_seconds'] = elapsed
                        for call in (delta.tool_calls or []):
                            tool_delta_count += 1
                            if len(tool_delta_sample)<64:
                                tool_delta_sample.append(dict(index=call.index, id_present=bool(call.id),
                                    name_fragment_chars=len(call.function.name or '') if call.function else 0,
                                    argument_fragment_chars=len(call.function.arguments or '') if call.function else 0,
                                    elapsed_seconds=elapsed))
                            item = tool_calls.setdefault(call.index, dict(
                                id='', type='function', function=dict(name='', arguments='')))
                            if call.id:
                                item['id'] = call.id
                            if call.function:
                                item['function']['name'] += call.function.name or ''
                                item['function']['arguments'] += call.function.arguments or ''
                            if allow_tool_calls and parameters.get('parallel_tool_calls') is False and len(tool_calls)>1:
                                failure_kind = 'multiple_tool_calls_when_disabled'
                                raise ValueError('Multiple native tool indices in a single-call request')
                        finish = choice.finish_reason or finish
                        if finish == 'length':
                            failure_kind = 'output_length_exceeded'
                            raise ValueError('Output truncated before a valid completed response')
                    save()
                    # The socket read timeout separately bounds stalled streams.
                    if time.monotonic() > overall_deadline:
                        failure_kind = 'stream_wall_time_exceeded'
                        raise TimeoutError('Stream wall time exceeded')
                phase = 'validating_response'
                if finish is None:
                    raise ValueError('Stream ended without a finish reason')
                # Some compatible gateways return complete native tool calls with stop.
                valid_tools = bool(allow_tool_calls and tool_calls and finish in {'tool_calls', 'stop'})
                if tool_calls and not valid_tools:
                    raise ValueError('Unexpected or incomplete tool calls')
                message = dict(role='assistant', content=content, reasoning_content=reasoning)
                if valid_tools:
                    calls = [tool_calls[i] for i in sorted(tool_calls)]
                    if any(not c['id'] or not c['function']['name'] for c in calls):
                        raise ValueError('Incomplete tool identity')
                    if 'tool_choice' in adjustments and isinstance(requested_parameters.get('tool_choice'), dict):
                        allowed = {tool['function']['name'] for tool in payload['tools']}
                        if any(call['function']['name'] not in allowed for call in calls):
                            failure_kind = 'unadvertised_tool_call'
                            raise ValueError('Response called a tool outside the declared set')
                    message['tool_calls'] = calls
                safe = redact(dict(id=response_id, model=returned_model, usage=usage,
                    archive_format='normalized_stream_response', choices=[dict(index=0,
                    finish_reason=finish, message=message)]), [key])
                atomic_json(attempt / 'response.json', safe)
                (attempt / 'response.txt').write_text(safe['choices'][0]['message']['content'])
                (attempt / 'reasoning.txt').write_text(safe['choices'][0]['message']['reasoning_content'])
                if usage is not None and usage.get('prompt_tokens') is not None:
                    metrics.update(prompt_tokens=usage['prompt_tokens'], token_count_source='provider_usage')
                    atomic_json(attempt / 'input-metrics.json', metrics)
                atomic_json(attempt / 'result.json', dict(
                    status='received' if content.strip() or valid_tools else 'empty', finished_at=utc(),
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    finish_reason=finish, timing=timings))
                if not content.strip() and not valid_tools:
                    raise ValueError('Stream returned no final answer')
                audit.finish(response=safe, usage=usage, metrics=dict(timing=timings, chunks=chunks_received))
                return safe['choices'][0]['message']['content'], attempt
            except BaseException as error:
                audit.finish(response=dict(content=content, tool_calls=list(tool_calls.values())),
                             usage=usage, error=error, metrics=dict(timing=timings, chunks=chunks_received))
                if not isinstance(error, Exception):
                    raise
                if isinstance(error, NoOutputTimeout):
                    failure_kind = 'no_output_timeout'
                elif isinstance(error, RequestDeadlineExceeded):
                    failure_kind = 'request_wall_time_exceeded'
                elif failure_kind is None and (isinstance(error,TimeoutError)
                        or type(error).__name__=='APITimeoutError'):
                    failure_kind = 'transport_timeout'
                public_error=StreamingRequestFailure('Streaming request failed; see archived attempt',
                    failure_kind=failure_kind,error_type=type(error).__name__,phase=phase,archive_ref=attempt,
                    output_received=output_received,chunks_received=chunks_received)
                timeout_detail=request_timeout_details(public_error)
                save()
                # Types/errno distinguish connection, header waits and broken streams
                # without persisting exception messages, URLs, headers or bodies.
                causes = []
                cause = error
                seen = set()
                while cause is not None and id(cause) not in seen and len(causes) < 8:
                    seen.add(id(cause))
                    item = dict(type=type(cause).__name__)
                    errno = getattr(cause, 'errno', None)
                    if isinstance(errno, int):
                        item['errno'] = errno
                    causes.append(item)
                    cause = cause.__cause__ or cause.__context__
                # Classify known parameter failures without recording gateway messages.
                body = getattr(error, 'body', None)
                diagnostic = json.dumps(body, ensure_ascii=False).lower() if body else ''
                provider_error = body.get('error', body) if isinstance(body, dict) else {}
                safe_provider = {}
                if isinstance(provider_error, dict):
                    for field in ('code', 'type', 'param'):
                        value = provider_error.get(field)
                        if isinstance(value, (str, int, float)) and len(str(value)) <= 200:
                            safe_provider[field] = value
                markers = [term for term in (
                    'max_completion_tokens', 'max_tokens', 'thinking_budget', 'enable_thinking',
                    'stream_options', 'stream', 'unsupported', 'not supported', 'minimum',
                    'maximum', 'greater', 'less', 'invalid', 'unknown', 'unrecognized',
                    'reasoning_effort', 'budget') if term in diagnostic]
                atomic_json(attempt / 'error.json', redact(dict(
                    error_type=type(error).__name__, http_status=getattr(error, 'status_code', None),
                    failure_phase=phase, failure_kind=failure_kind,
                    timeout_category=None if timeout_detail is None else timeout_detail['timeout_category'],
                    exception_chain=causes, chunks_received=chunks_received,
                    output_received=output_received,
                    request_total_seconds=total_seconds, no_output_seconds=no_output_seconds,
                    timing=timings, finished_at=utc(),
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    partial_response_saved=True, parameter_error_markers=markers,
                    provider_error=safe_provider), [key]))
                remaining = overall_deadline - time.monotonic()
                if (attempt_number < max_attempts and not output_received
                        and self._is_retryable_zero_output(error) and remaining > 1.0):
                    delay = retry_backoff_s[min(attempt_number - 1, len(retry_backoff_s) - 1)] if retry_backoff_s else 0.0
                    time.sleep(min(delay, max(0.0, remaining - 1.0)))
                    continue
                raise public_error from error
