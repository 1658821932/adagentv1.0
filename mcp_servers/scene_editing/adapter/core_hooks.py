"""AdAgent capabilities supplied to the standalone editor by the adapter."""
import json
import os

from common.llm_audit import RequestAudit, audited_json_call, record_operation, record_tool_exchange
from mcp_servers.process import PROJECT_ROOT
from mcp_servers.worldengine.adapter import planner_prewarm
from mcp_servers.worldengine.adapter.evaluation_horizon import DEFAULT_NATIVE_FRAMES, DEFAULT_NATIVE_HISTORY


def native_policy_horizon():
    return DEFAULT_NATIVE_HISTORY, DEFAULT_NATIVE_FRAMES


def start_prewarm(policy, report):
    return planner_prewarm.start(policy, report)


def native_policy_probe(policy, request, log):
    from mcp_servers.process import run_logged

    env = dict(os.environ, WORLDENGINE_ROOT=policy['worldengine_root'])
    warm = planner_prewarm.locate(policy)
    if warm is not None:
        env['ADAGENT_VAD_PREWARM_SOCKET'] = warm['socket_path']
    completed = run_logged([policy['python'], '-m', 'mcp_servers.worldengine.adapter.policy_probe_runner'],
        input=json.dumps(request), cwd=PROJECT_ROOT, env=env, log_path=log,
        json_stdout=True, timeout=request['timeout'] + 60)
    if completed.returncode:
        raise RuntimeError('Native VADv2 self-test failed; inspect ' + str(log))
    return json.loads(completed.stdout)
