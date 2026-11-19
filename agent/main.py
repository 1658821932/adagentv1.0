"""
Agent execution entry.

This module connects runtime configuration, the LLM orchestrator, Tool
executor, Agent state and structured run logger. It implements the minimal
Agent loop:

Task -> LLM decision -> Tool execution -> State update -> Next decision
"""

import sys
from typing import Any, Dict, Optional

from mcp_servers.runtime import trace_context
from agent import prewarm
from agent.config import AgentConfig
from agent.executor import ToolExecutor
from agent.logger import RunLogger
from agent.orchestrator import Orchestrator
from agent.state import AgentState
from agent.continuation import recover
from mcp_servers.scene_editing.core.adversarial.llm_request_policy import effective_reasoning_effort


class Agent:
    """Minimal autonomous driving data generation agent runtime."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig.load()
        self.orchestrator = Orchestrator(
            decision_correction_attempts=(
                self.config.decision_correction_attempts
            )
        )
        self.executor = ToolExecutor(
            tool_modes=self.config.tool_modes
        )
        self.state = None
        self.logger = None

    def run(
        self,
        task: str,
        max_steps: Optional[int] = None,
        *, resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute until completion, explicit stop, or the global step limit."""

        run_max_steps = (
            self.config.max_steps
            if max_steps is None
            else max_steps
        )
        if (
            not isinstance(run_max_steps, int)
            or isinstance(run_max_steps, bool)
            or run_max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")

        pending = None
        lineage = None
        if resume_from is not None:
            self.state, lineage, pending = recover(self.config.logs_root, resume_from, task)
        else:
            self.state = AgentState(task)
        # Validate the parent before starting any background preparation.
        prewarm.start()
        first_step = self.state.current_step + 1 if resume_from else 0
        self.logger = RunLogger(
            task=task,
            max_steps=run_max_steps,
            llm_model=self.orchestrator.llm_client.model,
            llm_base_url=self.orchestrator.llm_client.base_url,
            tool_modes=self.config.tool_modes,
            llm_reasoning_effort=effective_reasoning_effort(getattr(self.orchestrator.llm_client, 'request_profile', {})),
            logs_root=self.config.logs_root,
            lineage=lineage,
        )

        self.state.execution_scope = self.state.execution_scope or self.logger.run_id
        self.executor.scope = self.state.execution_scope
        self.logger._write_json(self.logger.run_dir/'initial_state.json', self.state.to_dict())
        self.logger.save_state(self.state.to_dict())
        # 将模型流式输出接到当前运行的日志，供控制台实时展示思考过程
        try:
            self.orchestrator.llm_client.on_delta = self.logger.thinking_delta
        except Exception:
            pass

        print("=" * 60)
        print("Agent Start")
        print("Task:")
        print(task)
        print("Logs:")
        print(self.logger.run_dir)
        print("=" * 60)

        try:
            for step in range(first_step, first_step + run_max_steps):
                self.state.set_step(step)

                print(f"\n===== Agent Step {step} =====")

                recovering = pending is not None
                if recovering:
                    arguments = dict(pending['arguments'])
                    control = arguments.pop('_execution', {})
                    job_id = control.get('job_id') or (self.state.pending_job or {}).get('job_id')
                    if job_id is None:
                        job_id = self.executor.jobs.lookup(pending['tool_name'],
                            self.executor.tool_modes[pending['tool_name']], arguments, self.executor.scope)
                    if job_id is None:
                        raise RuntimeError('Unfinished parent call has no persistent job; refusing to repeat it automatically. '
                                           'Inspect the parent logs before explicitly retrying this tool.')
                    arguments['_execution'] = dict(job_id=job_id)
                    decision = dict(action='call_tool', tool_name=pending['tool_name'], arguments=arguments,
                                    reason='Reconnect to the previously submitted tool job and recover its real result.')
                    pending = None
                else:
                    self.logger.open_thinking(step)
                    try:
                        with trace_context(agent_run_id=self.logger.run_id,
                                           agent_log_dir=str(self.logger.run_dir.resolve()),
                                           stage='main_agent', step=step):
                            decision = self.orchestrator.decide(
                                self.state.task,
                                self.state.to_orchestrator_dict(),
                            )
                    finally:
                        self.logger.close_thinking()

                print("LLM Decision:")
                print(decision)

                self.state.add_decision(
                    step,
                    decision,
                )
                self.logger.log_decision(
                    step,
                    decision,
                    request_meta={"recovered_pending_job": True} if recovering else {
                        "metrics":getattr(self.orchestrator.llm_client,"last_request_metrics",{}),
                        "usage":getattr(self.orchestrator.llm_client,"last_usage",None),
                    },
                )

                if decision.get("action") == "finish":
                    if self.state.background_jobs:
                        self.state.error='Collect the existing background jobs before finishing; use their job_id without retry.'
                        self.logger.save_state(self.state.to_dict())
                        continue
                    self.state.mark_finished()
                    final_state = self.state.to_dict()
                    self.logger.finalize(final_state)
                    print("Agent finished.")
                    return final_state

                if decision.get("action") == "stop":
                    self.state.mark_error(decision["reason"])
                    final_state = self.state.to_dict()
                    self.logger.finalize(final_state)
                    print("Agent stopped without completing the task.")
                    return final_state

                if decision.get("action") != "call_tool":
                    self.state.mark_error(decision)
                    final_state = self.state.to_dict()
                    self.logger.finalize(final_state)
                    print("Invalid action returned by LLM.")
                    return final_state

                tool_name = decision.get("tool_name")
                arguments = decision.get("arguments", {})

                self.state.set_stage_from_tool(tool_name)

                print("Executing Tool:")
                print(tool_name)
                print("Arguments:")
                print(arguments)

                self.logger.start_tool_call(
                    step,
                    tool_name,
                    arguments,
                )
                self.logger.save_state(self.state.to_dict())

                try:
                    with trace_context(agent_run_id=self.logger.run_id,
                                       agent_log_dir=str(self.logger.run_dir.resolve()),
                                       step=step, stage=tool_name,
                                       parent_run_id=self.logger.run_id,
                                       tool_call_id=f"{self.logger.run_id}:step_{step:02d}:{tool_name}"):
                        result = self.executor.execute(tool_name, arguments)
                except Exception as exc:
                    self.logger.log_tool_error(
                        step,
                        tool_name,
                        exc,
                    )
                    raise

                background=arguments.get('_execution',{}).get('background',False)
                while (result.get("status") == "running" and not background
                       and (result.get('job') or {}).get('state') != 'orphaned'):
                    self.state.pending_job = result["job"]
                    self.logger.save_state(self.state.to_dict())
                    print("Tool still running; waiting for existing job:", result["job"]["job_id"])
                    result = self.executor.query(
                        result["job"]["job_id"],
                        wait_timeout=self.executor.execution_timeout,
                    )
                orphaned = (result.get('job') or {}).get('state') == 'orphaned'
                if orphaned:
                    job = result['job']
                    message = f"Tool job {job['job_id']} lost its worker; a registered child remains. "
                    message += "Inspect the saved job before retrying."
                    result = dict(result, status='error', error=dict(type='job_orphaned', message=message))
                self.state.pending_job = None
                job_id = (result.get('job') or {}).get('job_id')
                if job_id:
                    requested_job_id = arguments.get('_execution', {}).get('job_id')
                    if requested_job_id and requested_job_id != job_id:
                        self.state.background_jobs.pop(requested_job_id, None)
                    if result.get('status') == 'running':
                        self.state.background_jobs[job_id] = dict(tool_name=tool_name, job_id=job_id,
                            arguments={k: v for k, v in arguments.items() if k != '_execution'})
                    else:
                        self.state.background_jobs.pop(job_id, None)

                result_ref = self.logger.finish_tool_call(
                    step,
                    tool_name,
                    result,
                )

                print("Tool Result:")
                print(result)

                self.state.add_tool_result(tool_name, result, result_ref=result_ref)
                self.logger.save_state(self.state.to_dict())
                if orphaned:
                    self.state.mark_error(message)
                    final_state = self.state.to_dict()
                    self.logger.finalize(final_state)
                    return final_state

            self.state.mark_error("Maximum agent steps reached")
            final_state = self.state.to_dict()
            self.logger.finalize(final_state)
            return final_state

        except KeyboardInterrupt:
            self.state.mark_error("Interrupted by user")
            self.logger.finalize(self.state.to_dict())
            raise
        except Exception as exc:
            self.state.mark_error(
                f"{type(exc).__name__}: {exc}"
            )
            self.logger.finalize(self.state.to_dict())
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run a task, optionally continuing an existing conversation.')
    parser.add_argument('--resume-from', help='Parent run ID under the configured logs directory')
    parser.add_argument('task', nargs='*')
    args = parser.parse_args()
    agent = Agent()

    if args.task:
        task = " ".join(args.task)
    else:
        task = "根据自动驾驶失败案例分析原因，并设计一个新的测试场景。"

    result = agent.run(task, resume_from=args.resume_from)

    print("\nFinal State:")
    print(result)
