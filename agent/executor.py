"""Common persistent execution, status queries and duplicate-start prevention."""
from uuid import uuid4
from agent.jobs import JobStore
from tools.registry import DEFAULT_TOOL_MODES

DEFAULT_EXECUTION_TIMEOUT = 720

class ToolExecutor:
    def __init__(self, tool_modes=None, execution_timeout=DEFAULT_EXECUTION_TIMEOUT, *, jobs_root=None, scope=None):
        self.tool_modes = dict(tool_modes or DEFAULT_TOOL_MODES)
        self.execution_timeout = execution_timeout
        self.jobs = JobStore(jobs_root)
        self.scope = scope or uuid4().hex

    def query(self, job_id, *, wait_timeout=0):
        return self.jobs.wait(job_id, wait_timeout)

    def execute(self, tool_name, arguments):
        try:
            mode = self.tool_modes[tool_name]
            arguments = dict(arguments)
            control = arguments.pop("_execution", {})
            if not isinstance(control, dict) or set(control) - {"job_id", "retry", "background"}:
                raise ValueError("_execution accepts only job_id, retry and background")
            if 'background' in control and type(control['background']) is not bool:
                raise ValueError('_execution.background must be boolean')
            if control.get('background') and tool_name != 'policy_runner':
                raise ValueError('Background execution is supported for independent 06/07 branches')
            if "retry" in control and type(control["retry"]) is not bool:
                raise ValueError("_execution.retry must be boolean")
            if control.get("job_id"):
                manifest = self.jobs._folder(control["job_id"]) / "manifest.json"
                import json
                job = json.loads(manifest.read_text())
                if job["tool"] != tool_name or job["mode"] != mode or job["scope"] != self.scope:
                    raise ValueError("job_id does not belong to this tool/mode/run")
                if control.get("retry"):
                    request = json.loads((manifest.parent / "request.json").read_text())
                    job_id = self.jobs.submit(tool_name, mode, request["arguments"], self.scope, retry=True)
                else:
                    job_id = control["job_id"]
            else:
                job_id = self.jobs.submit(tool_name, mode, arguments, self.scope, retry=control.get("retry", False))
            return self.jobs.wait(job_id, 0 if control.get('background') else self.execution_timeout)
        except Exception as exc:
            return dict(status="error", tool=tool_name,
                        error=dict(type="execution_error", message=str(exc)))
