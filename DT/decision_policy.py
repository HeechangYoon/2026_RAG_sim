from typing import Any

from DT.agents import BaseSchedulingAgent, OpenAISchedulingAgent, ReinforcementLearningAgent


class BaseDecisionPolicy:
    """Hook interface consumed by the simulator components."""

    def bind_runtime(self, env, monitor, model, resource) -> None:
        self.env = env
        self.monitor = monitor
        self.model = model
        self.resource = resource

    def choose_sequence(
        self,
        state: dict[str, Any],
        candidate_job_ids: list[str],
        default_order: list[str],
    ) -> list[str]:
        return default_order

    def choose_routing(
        self,
        state: dict[str, Any],
        candidate_machine_types: list[str],
        default_choice: str | None,
    ) -> str | None:
        return default_choice

    def choose_dispatch(
        self,
        state: dict[str, Any],
        candidate_job_ids: list[str],
        default_choice: str | None,
    ) -> str | None:
        return default_choice


class AgentDecisionPolicy(BaseDecisionPolicy):
    """
    Adapter that lets the simulator use a generic agent.
    The agent only needs a single `act(...)` method.
    """

    def __init__(self, agent: BaseSchedulingAgent):
        self.agent = agent

    def bind_runtime(self, env, monitor, model, resource) -> None:
        super().bind_runtime(env, monitor, model, resource)
        if hasattr(self.agent, "bind_runtime"):
            self.agent.bind_runtime(env=env, monitor=monitor, model=model, resource=resource)

    def choose_sequence(
        self,
        state: dict[str, Any],
        candidate_job_ids: list[str],
        default_order: list[str],
    ) -> list[str]:
        if not candidate_job_ids:
            return default_order
        default_choice = default_order[0] if default_order else candidate_job_ids[0]
        selected = self.agent.act(
            decision_type="sequencing",
            state=state,
            candidates=candidate_job_ids,
            default_action=default_choice,
        )
        if selected is None:
            return default_order
        remaining = [job_id for job_id in default_order if job_id != selected]
        return [selected] + remaining

    def choose_routing(
        self,
        state: dict[str, Any],
        candidate_machine_types: list[str],
        default_choice: str | None,
    ) -> str | None:
        return self.agent.act(
            decision_type="routing",
            state=state,
            candidates=candidate_machine_types,
            default_action=default_choice,
        )

    def choose_dispatch(
        self,
        state: dict[str, Any],
        candidate_job_ids: list[str],
        default_choice: str | None,
    ) -> str | None:
        return self.agent.act(
            decision_type="dispatch",
            state=state,
            candidates=candidate_job_ids,
            default_action=default_choice,
        )


class OpenAIDecisionPolicy(AgentDecisionPolicy):
    """Backward-compatible wrapper around the OpenAI-compatible scheduling agent."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 250,
        timeout_sec: int = 30,
        rag_db_path: str | None = None,
        rag_top_k: int = 2,
        use_rag: bool = True,
    ):
        super().__init__(
            OpenAISchedulingAgent(
                api_url=api_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                rag_db_path=rag_db_path,
                rag_top_k=rag_top_k,
                use_rag=use_rag,
            )
        )


class ReinforcementLearningDecisionPolicy(AgentDecisionPolicy):
    """Adapter for RL agents or policy networks."""

    def __init__(self, policy_model):
        super().__init__(ReinforcementLearningAgent(policy_model=policy_model))
