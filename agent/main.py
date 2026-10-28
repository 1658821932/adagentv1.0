"""
Agent execution entry.

This module connects runtime configuration, the LLM orchestrator, Tool
executor, Agent state and structured run logger. It implements the minimal
Agent loop:

Task -> LLM decision -> Tool execution -> State update -> Next decision
"""

import sys
from typing import Any, Dict, Optional

from agent.config import AgentConfig
from agent.executor import ToolExecutor
from agent.logger import RunLogger
from agent.orchestrator import Orchestrator
from agent.state import AgentState


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
            tool_modes=self.config.tool_modes,
            reconstruction_backend=self.config.reconstruction_backend,
        )
        self.state = None
        self.logger = None

    def run(
        self,
        task: str,
        max_steps: Optional[int] = None,
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

        self.executor.reset()
        self.state = AgentState(task)
        self.logger = RunLogger(
            task=task,
            max_steps=run_max_steps,
            llm_model=self.orchestrator.llm_client.model,
            llm_base_url=self.orchestrator.llm_client.base_url,
            tool_modes=self.config.tool_modes,
            logs_root=self.config.logs_root,
        )

        print("=" * 60)
        print("Agent Start")
        print("Task:")
        print(task)
        print("Logs:")
        print(self.logger.run_dir)
        print("=" * 60)

        try:
            for step in range(run_max_steps):
                self.state.set_step(step)

                print(f"\n===== Agent Step {step} =====")

                decision = self.orchestrator.decide(
                    task,
                    {**self.state.to_orchestrator_dict(),
                     "reconstruction_backend": self.config.reconstruction_backend},
                )

                print("LLM Decision:")
                print(decision)

                self.state.add_decision(
                    step,
                    decision,
                )
                self.logger.log_decision(
                    step,
                    decision,
                )

                if decision.get("action") == "finish":
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

                try:
                    result = self.executor.execute(
                        tool_name,
                        arguments,
                    )
                except Exception as exc:
                    self.logger.log_tool_error(
                        step,
                        tool_name,
                        exc,
                    )
                    raise

                self.logger.finish_tool_call(
                    step,
                    tool_name,
                    result,
                )

                print("Tool Result:")
                print(result)

                self.state.add_tool_result(
                    tool_name,
                    result,
                )

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
    agent = Agent()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "根据自动驾驶失败案例分析原因，并设计一个新的测试场景。"

    result = agent.run(task)

    print("\nFinal State:")
    print(result)
