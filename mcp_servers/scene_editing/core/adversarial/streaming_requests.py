"""Archive streamed LLM reasoning and answers, including interrupted responses."""
import json
import time
import uuid
import queue
import threading

from experiment_runtime import RequestArchive, atomic_json, redact, utc
from kimi_request_parameters import normalize_kimi_parameters
from model_context import input_metrics


class RequestDeadlineExceeded(TimeoutError):
    pass


def deadline_events(create_stream, seconds):
    """Bound header waits and idle reads together. Only caller writes archives.

    Network I/O runs in a daemon; on timeout close the response asynchronously
    so a blocked socket close cannot hold up the caller. Transport timeout also
    bounds an outstanding header request whose response is not available yet.
    """
    events = queue.Queue(maxsize=8)
    stopped = threading.Event()
    opened = []
    deadline = time.monotonic()+seconds

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
            remaining = deadline-time.monotonic()
            if remaining<=0:
                raise RequestDeadlineExceeded("Request wall-clock deadline exceeded")
            try:
                kind,value = events.get(timeout=remaining)
            except queue.Empty:
                raise RequestDeadlineExceeded("Request wall-clock deadline exceeded") from None
            if time.monotonic()>=deadline:
                raise RequestDeadlineExceeded("Request wall-clock deadline exceeded")
            if kind=="error":
                raise value
            if kind=="end":
                return
            yield kind,value
    finally:
        stopped.set()
        if opened:
            threading.Thread(target=close,daemon=True).start()



class StreamingRequestArchive(RequestArchive):
    def complete(self, label, messages, *, total_seconds=300, allow_tool_calls=False, **parameters):
        if type(total_seconds) not in (int,float) or not 0 < total_seconds < float('inf'):
            raise ValueError('total_seconds must be positive and finite')
        key = self.profile['api_key']
        if key and key in json.dumps(messages, ensure_ascii=False):
            raise ValueError('Credential found in outgoing message')
        requested_parameters = parameters
        parameters = normalize_kimi_parameters(self.profile['model'], parameters)
        adjustments = [field for field in sorted(requested_parameters.keys() | parameters.keys())
            if requested_parameters.get(field) != parameters.get(field)]
        attempt = self.folder / (label + '_' + uuid.uuid4().hex) / 'attempt_001'
        attempt.mkdir(parents=True, exist_ok=False)
        payload = dict(model=self.profile['model'], messages=messages, **parameters,
                       stream=True, stream_options={'include_usage': True})
        atomic_json(attempt / 'request.json', redact(dict(
            started_at=utc(), endpoint=self.profile['base_url'], payload=payload,
            parameter_adjustments=adjustments), [key]))
        metrics = input_metrics(messages, parameters.get('tools', ()))
        atomic_json(attempt / 'input-metrics.json', metrics)
        started = time.monotonic()
        reasoning = content = ''
        usage = None
        finish = None
        returned_model = None
        response_id = None
        timings = {}
        tool_calls = {}
        phase = 'waiting_response_headers'
        chunks_received = 0
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
                    lambda: self.client.chat.completions.create(**payload, timeout=total_seconds),
                    max(0., total_seconds-(time.monotonic()-started))):
                if event == 'headers':
                    timings['response_headers_seconds'] = round(time.monotonic()-started,3)
                    phase = 'reading_response_stream'
                    continue
                chunks_received += 1
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
                        timings.setdefault('first_reasoning_seconds', elapsed)
                        timings['last_reasoning_seconds'] = elapsed
                    if answer:
                        timings.setdefault('first_answer_seconds', elapsed)
                        timings['last_answer_seconds'] = elapsed
                    reasoning += thought
                    content += answer
                    if delta.tool_calls:
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
                if elapsed > total_seconds:
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
            return safe['choices'][0]['message']['content'], attempt
        except Exception as error:
            if isinstance(error, RequestDeadlineExceeded):
                failure_kind = 'request_wall_time_exceeded'
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
                exception_chain=causes, chunks_received=chunks_received,
                request_total_seconds=total_seconds, timing=timings,
                finished_at=utc(), elapsed_seconds=round(time.monotonic() - started, 3),
                partial_response_saved=True, parameter_error_markers=markers,
                provider_error=safe_provider), [key]))
            raise RuntimeError('Streaming request failed; see archived attempt') from None
