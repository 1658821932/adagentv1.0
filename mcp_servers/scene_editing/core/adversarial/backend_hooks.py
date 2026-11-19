"""Optional host capabilities for the standalone scene editor."""
from importlib import import_module
import os


def _host():
    name = os.getenv('ADVERSARIAL_HOST_HOOKS')
    return import_module(name) if name else None


class UnrecordedRequest:
    def first_response(self):
        pass

    def finish(self, **result):
        pass


def request_audit(*args, **kwargs):
    host = _host()
    return host.RequestAudit(*args, **kwargs) if host else UnrecordedRequest()


def audited_json_call(call, payload, **options):
    host = _host()
    if host:
        return host.audited_json_call(call, payload, **options)
    response = call()
    body = response.json() if not hasattr(response, 'model_dump') and hasattr(response, 'json') else response
    return (response, body) if options.get('return_response') else body


def record_tool_exchange(*args, **kwargs):
    host = _host()
    if host:
        host.record_tool_exchange(*args, **kwargs)


def record_operation(*args, **kwargs):
    host = _host()
    if host:
        host.record_operation(*args, **kwargs)


def native_policy_horizon():
    host = _host()
    if host is None:
        raise RuntimeError('Native policy testing requires WorldEngine integration via ADVERSARIAL_HOST_HOOKS')
    return host.native_policy_horizon()


def start_prewarm(policy, report):
    host = _host()
    return host.start_prewarm(policy, report) if host else None


def native_policy_probe(policy, request, log):
    host = _host()
    if host is None:
        raise RuntimeError('Native policy testing requires WorldEngine integration via ADVERSARIAL_HOST_HOOKS')
    return host.native_policy_probe(policy, request, log)
