"""Bounded recovery for transport errors and upstream HTTP 5xx model responses."""


RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)
UPSTREAM_RETRY_DELAYS_SECONDS = (5.0, 15.0, 30.0)


def call_with_connection_retries(operation, *, metrics=None, sleep=None, notify=None):
    try:
        from openai import APIConnectionError, InternalServerError
        upstream_errors = (InternalServerError,)
        retryable_errors = (APIConnectionError, *upstream_errors)
    except ImportError:
        upstream_errors = retryable_errors = ()
    from mcp_servers.runtime import trace_context
    from time import sleep as default_sleep

    pause = sleep or default_sleep
    max_retries = len(RETRY_DELAYS_SECONDS)
    for attempt in range(max_retries + 1):
        try:
            with trace_context(llm_attempt=attempt + 1):
                return operation()
        except retryable_errors as error:
            chunks = metrics().get('chunks', 0) if metrics is not None else 0
            if chunks or attempt == max_retries:
                raise
            delays = (UPSTREAM_RETRY_DELAYS_SECONDS if isinstance(error, upstream_errors)
                      else RETRY_DELAYS_SECONDS)
            delay = delays[attempt]
            if notify is not None:
                notify(attempt + 1, delay, error)
            pause(delay)
