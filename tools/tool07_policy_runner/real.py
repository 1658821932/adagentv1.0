"""Agent tool contract for policy_runner; backend execution lives in its adapter."""

from mcp_servers.worldengine.adapter.service import run_policy, run_pdm


def run(input_data: dict) -> dict:
    if input_data.get('execution_mode') == 'pdm_closed_loop':
        try:
            result = run_pdm(input_data)
        except Exception as error:
            return {'status':'error','tool':'policy_runner','error':{'type':'pdm_challenge_failed','message':str(error)[:300]}}
        return {'status':'success','tool':'policy_runner','result':result}
    return {
        "status": "success",
        "tool": "policy_runner",
        "result": run_policy(input_data),
    }
