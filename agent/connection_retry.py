"""Bounded transport recovery for model requests; no business/tool replay."""


RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)


def call_with_connection_retries(operation, *, metrics=None, sleep=None, notify=None):
    try:
        from openai import APIConnectionError, APITimeoutError
        retryable_errors = (APIConnectionError, APITimeoutError)
    except ImportError:
        retryable_errors = ()
    from mcp_servers.runtime import trace_context
    from time import sleep as default_sleep

    pause = sleep or default_sleep
    delays = RETRY_DELAYS_SECONDS
    for attempt in range(len(delays) + 1):
        try:
            with trace_context(llm_attempt=attempt + 1):
                return operation()
        except retryable_errors as error:
            chunks = metrics().get('chunks', 0) if metrics is not None else 0
            if chunks or attempt == len(delays):
                raise
            delay = delays[attempt]
            if notify is not None:
                notify(attempt + 1, delay, error)
            pause(delay)
