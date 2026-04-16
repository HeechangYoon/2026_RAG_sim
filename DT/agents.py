import json
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from DT.utils.ortools_jssp import solve_jssp_schedule


class BaseSchedulingAgent:
    """Generic agent interface used by the simulator at each decision point."""

    def bind_runtime(self, env, monitor, model, resource) -> None:
        self.env = env
        self.monitor = monitor
        self.runtime_model = model
        self.runtime_resource = resource
        # Runtime binding should not overwrite agent-specific configuration
        # such as strict-failure mode or the decision trace logger.
        if not hasattr(self, "decision_logger"):
            self.decision_logger = None
        if not hasattr(self, "fail_on_invalid_action"):
            self.fail_on_invalid_action = False

    def act(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str | None:
        return default_action


class AgentDecisionError(RuntimeError):
    """Raised when an agent returns an unusable action and the run should stop."""


class DecisionTraceLogger:
    """JSONL logger for agent decision requests/responses."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        record = dict(payload)
        record["logged_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


class HeuristicSchedulingAgent(BaseSchedulingAgent):
    """
    Agent wrapper for existing heuristic defaults.
    It simply accepts the default action computed by the simulator.
    """

    def __init__(self, decision_log_path: str | None = None):
        self.decision_logger = DecisionTraceLogger(decision_log_path)
        self.fail_on_invalid_action = False

    def act(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str | None:
        if self.decision_logger is not None:
            self.decision_logger.log(
                {
                    "decision_type": decision_type,
                    "state": state,
                    "candidates": candidates,
                    "default_action": default_action,
                    "raw_response": "",
                    "parsed_choice": default_action,
                    "reason": "heuristic_default",
                    "error": None,
                    "query_error": None,
                    "usage": None,
                    "response_body": None,
                }
            )
        return default_action


class OptimalReplaySchedulingAgent(BaseSchedulingAgent):
    """
    Replays a solver-produced JSSP schedule inside the stepwise simulator.

    At each decision point, the agent selects the candidate whose operation appears
    next in the OR-Tools schedule. For routing, it picks the solver-assigned machine.
    """

    def __init__(
        self,
        problem_data: dict[str, Any],
        decision_log_path: str | None = None,
        time_limit_sec: float | None = None,
        require_optimal: bool = False,
    ):
        self.problem_data = problem_data
        self.schedule_plan = solve_jssp_schedule(
            problem_data=problem_data,
            time_limit_sec=time_limit_sec,
            require_optimal=require_optimal,
        )
        self.decision_logger = DecisionTraceLogger(decision_log_path)
        self.fail_on_invalid_action = True
        self._operations_by_id = self.schedule_plan["operations_by_id"]
        self._job_info = self.problem_data.get("job_info", {})

    def _operation_sort_key(self, operation_id: str) -> tuple[int, int, str]:
        row = self._operations_by_id.get(operation_id)
        if row is None:
            return (10**12, 10**12, operation_id)
        return (int(row.start), int(row.end), str(operation_id))

    def _candidate_operation_id(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidate: str,
    ) -> str | None:
        if decision_type == "dispatch":
            for row in state.get("candidate_jobs", []):
                if isinstance(row, dict) and str(row.get("job_id")) == candidate:
                    return str(row.get("operation_id") or "")
            return None

        if decision_type == "sequencing":
            job_info = self._job_info.get(candidate, {})
            operations = job_info.get("operations", [])
            return str(operations[0]) if operations else None

        if decision_type == "routing":
            return str(state.get("operation_id") or "")

        return None

    def _choose_replay_action(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str | None:
        if not candidates:
            return default_action

        if decision_type == "routing":
            operation_id = self._candidate_operation_id(decision_type, state, candidates[0])
            row = self._operations_by_id.get(operation_id or "")
            if row is not None and row.machine_type in candidates:
                return row.machine_type
            return default_action

        ranked_candidates: list[tuple[tuple[int, int, str], str]] = []
        for candidate in candidates:
            operation_id = self._candidate_operation_id(decision_type, state, candidate)
            if operation_id:
                ranked_candidates.append((self._operation_sort_key(operation_id), candidate))
        if ranked_candidates:
            ranked_candidates.sort(key=lambda item: (item[0], item[1]))
            return ranked_candidates[0][1]
        return default_action

    def act(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str | None:
        choice = self._choose_replay_action(decision_type, state, candidates, default_action)
        if self.decision_logger is not None:
            self.decision_logger.log(
                {
                    "decision_type": decision_type,
                    "state": state,
                    "candidates": candidates,
                    "default_action": default_action,
                    "raw_response": "",
                    "parsed_choice": choice,
                    "reason": "ortools_replay",
                    "error": None,
                    "query_error": None,
                    "usage": None,
                    "response_body": {
                        "solver_status": self.schedule_plan.get("status"),
                        "is_optimal": self.schedule_plan.get("is_optimal"),
                    },
                }
            )
        return choice


class SelectiveDecisionAgent(BaseSchedulingAgent):
    """
    Delegates only selected decision types to the wrapped agent.
    All other decision types immediately fall back to the simulator default.
    """

    def __init__(self, agent: BaseSchedulingAgent, decision_types: set[str] | None = None):
        self.agent = agent
        self.decision_types = {item.strip() for item in (decision_types or set()) if item and item.strip()}

    def bind_runtime(self, env, monitor, model, resource) -> None:
        super().bind_runtime(env, monitor, model, resource)
        if hasattr(self.agent, "bind_runtime"):
            self.agent.bind_runtime(env, monitor, model, resource)

    def act(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str | None:
        if self.decision_types and decision_type not in self.decision_types:
            return default_action
        return self.agent.act(decision_type, state, candidates, default_action)


class OpenAISchedulingAgent(BaseSchedulingAgent):
    """
    OpenAI-compatible agent.
    It receives the simulator state plus candidate actions and returns one action id.
    """

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
        decision_log_path: str | None = None,
        prompt_log_path: str | None = None,
        show_prompt: bool = False,
        fail_on_invalid_action: bool = True,
        adaptive_max_tokens: bool = False,
        adaptive_token_step: int = 200,
        adaptive_token_cap: int | None = None,
        response_mode: str = "verbose_reason",
        rag_explanatory_cases: bool = True,
        rag_prefilter_candidate_gap: int = 1,
        rag_conditional: bool = False,
        rag_ambiguity_threshold: float = 0.12,
        rag_bottleneck_threshold: float = 3.0,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.rag_db_path = rag_db_path
        self.rag_top_k = rag_top_k
        self.use_rag = use_rag
        self.decision_logger = DecisionTraceLogger(decision_log_path)
        self.prompt_logger = DecisionTraceLogger(prompt_log_path)
        self.show_prompt = show_prompt
        self.fail_on_invalid_action = fail_on_invalid_action
        self.adaptive_max_tokens = adaptive_max_tokens
        self.adaptive_token_step = adaptive_token_step
        self.adaptive_token_cap = adaptive_token_cap
        self.response_mode = response_mode
        self.rag_explanatory_cases = rag_explanatory_cases
        self.rag_prefilter_candidate_gap = max(0, int(rag_prefilter_candidate_gap))
        self.rag_conditional = rag_conditional
        self.rag_ambiguity_threshold = float(rag_ambiguity_threshold)
        self.rag_bottleneck_threshold = float(rag_bottleneck_threshold)
        self.last_query_error = None
        self.last_response_body = None
        self.last_usage = None
        self.transient_retry_statuses = {502, 503, 504, 520}
        self.max_transient_retries = 3

    @staticmethod
    def _safe_json_loads(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _memory_rank_key(row: tuple[Any, ...], preferred_instance: str = "") -> tuple[int, float, str]:
        metadata = OpenAISchedulingAgent._safe_json_loads(str(row[3])) if len(row) > 3 else {}
        makespan = metadata.get("makespan")
        try:
            makespan_value = float(makespan)
        except Exception:
            makespan_value = float("inf")
        instance_name = str(row[0])
        same_instance_penalty = 0 if preferred_instance and instance_name == preferred_instance else 1
        return (same_instance_penalty, makespan_value, instance_name)

    @staticmethod
    def _state_instance_features(state: dict[str, Any]) -> dict[str, float]:
        problem_summary = state.get("problem_summary", {}) if isinstance(state.get("problem_summary"), dict) else {}
        machine_status = state.get("machine_status", []) if isinstance(state.get("machine_status"), list) else []
        global_jobs = state.get("global_job_summary", []) if isinstance(state.get("global_job_summary"), list) else []

        remaining_works = [float(row.get("remaining_work", 0.0) or 0.0) for row in global_jobs if isinstance(row, dict)]
        queue_sizes = []
        for row in machine_status:
            if not isinstance(row, dict):
                continue
            queued_jobs = row.get("queued_jobs")
            if isinstance(queued_jobs, list):
                queue_sizes.append(float(len(queued_jobs)))
            else:
                queue_sizes.append(float(row.get("queue_size", 0.0) or 0.0))
        max_queue = max(queue_sizes) if queue_sizes else 0.0
        avg_remaining = sum(remaining_works) / len(remaining_works) if remaining_works else 0.0
        return {
            "n_jobs": float(problem_summary.get("total_jobs", 0) or 0),
            "n_machines": float(len(machine_status)),
            "remaining_jobs": float(problem_summary.get("remaining_jobs", 0) or 0),
            "avg_job_work": float(avg_remaining),
            "bottleneck_utilization": float(max_queue),
        }

    @staticmethod
    def _candidate_set_features(
        candidate_jobs: list[dict[str, Any]],
        default_action: str | None = None,
    ) -> dict[str, Any]:
        proc_times: list[float] = []
        remaining_work: list[float] = []
        delta_makespans: list[float] = []
        critical_flags: list[float] = []
        candidate_profiles: list[tuple[float, float]] = []
        default_action_str = str(default_action or "")
        default_profile: dict[str, float] = {}

        for row in candidate_jobs:
            if not isinstance(row, dict):
                continue
            proc = float(row.get("proc_time_on_machine", 0.0) or 0.0)
            rem = float(row.get("remaining_work", 0.0) or 0.0)
            delta = float(row.get("delta_makespan", 0.0) or 0.0)
            critical = float(row.get("criticality_flag", 0.0) or 0.0)
            proc_times.append(proc)
            remaining_work.append(rem)
            delta_makespans.append(delta)
            critical_flags.append(critical)
            candidate_profiles.append((proc, rem))
            if default_action_str and str(row.get("job_id", "")) == default_action_str:
                post_route = row.get("post_route") if isinstance(row.get("post_route"), dict) else {}
                default_profile = {
                    "default_proc_time": proc,
                    "default_remaining_work": rem,
                    "default_delta_makespan": delta,
                    "default_criticality": critical,
                    "default_queue_after": float(row.get("affected_machine_load_after", 0.0) or 0.0),
                    "default_next_queue": float(post_route.get("next_stage_queue_pressure", 0.0) or 0.0),
                }

        def _top2_gap(values: list[float]) -> float:
            if len(values) < 2:
                return 0.0
            ordered = sorted(float(v) for v in values)
            return float(ordered[1] - ordered[0])

        features: dict[str, Any] = {
            "avg_proc_time": float(sum(proc_times) / len(proc_times)) if proc_times else 0.0,
            "avg_remaining_work": float(sum(remaining_work) / len(remaining_work)) if remaining_work else 0.0,
            "avg_delta_makespan": float(sum(delta_makespans) / len(delta_makespans)) if delta_makespans else 0.0,
            "min_proc_time": float(min(proc_times)) if proc_times else 0.0,
            "max_proc_time": float(max(proc_times)) if proc_times else 0.0,
            "min_remaining_work": float(min(remaining_work)) if remaining_work else 0.0,
            "max_remaining_work": float(max(remaining_work)) if remaining_work else 0.0,
            "min_delta_makespan": float(min(delta_makespans)) if delta_makespans else 0.0,
            "max_delta_makespan": float(max(delta_makespans)) if delta_makespans else 0.0,
            "critical_candidate_ratio": (
                float(sum(critical_flags) / len(critical_flags)) if critical_flags else 0.0
            ),
            "proc_gap_12": _top2_gap(proc_times),
            "rem_gap_12": _top2_gap(remaining_work),
            "delta_gap_12": _top2_gap(delta_makespans),
            "candidate_profile_vector": sorted(candidate_profiles, key=lambda item: (item[0], item[1])),
        }
        features.update(default_profile)
        return features

    @staticmethod
    def _state_decision_features(state: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
        candidate_jobs = state.get("candidate_jobs", []) if isinstance(state.get("candidate_jobs"), list) else []
        candidate_stats = OpenAISchedulingAgent._candidate_set_features(
            candidate_jobs,
            state.get("default_action"),
        )
        global_jobs = state.get("global_job_summary", []) if isinstance(state.get("global_job_summary"), list) else []
        global_remaining = [
            float(row.get("remaining_work", 0.0) or 0.0)
            for row in global_jobs
            if isinstance(row, dict)
        ]
        bottlenecks = state.get("bottleneck_machines", []) if isinstance(state.get("bottleneck_machines"), list) else []
        bottleneck_queue = 0.0
        if bottlenecks:
            try:
                bottleneck_queue = float(max(float(row.get("queue_size", 0.0) or 0.0) for row in bottlenecks if isinstance(row, dict)))
            except Exception:
                bottleneck_queue = 0.0
        return {
            "candidate_count": float(len(candidates)),
            "bottleneck_queue": bottleneck_queue,
            "global_remaining_work_vector": sorted(global_remaining, reverse=True)[:10],
            "global_remaining_jobs": float(len(global_remaining)),
            "global_avg_remaining_work": float(sum(global_remaining) / len(global_remaining)) if global_remaining else 0.0,
            "global_max_remaining_work": float(max(global_remaining)) if global_remaining else 0.0,
            "current_makespan": float(state.get("current_makespan", 0.0) or 0.0),
            "machine_ready_time": float(state.get("machine_ready_time", 0.0) or 0.0),
            **candidate_stats,
        }

    @staticmethod
    def _feature_distance(lhs: dict[str, float], rhs: dict[str, Any], keys: list[str]) -> float:
        distance = 0.0
        for key in keys:
            try:
                left_value = float(lhs.get(key, 0.0) or 0.0)
                right_value = float(rhs.get(key, 0.0) or 0.0)
            except Exception:
                continue
            scale = max(abs(left_value), abs(right_value), 1.0)
            distance += abs(left_value - right_value) / scale
        return distance

    @staticmethod
    def _vector_distance(lhs: list[float], rhs: list[Any]) -> float:
        left = [float(v) for v in lhs]
        right = []
        for value in rhs:
            try:
                right.append(float(value))
            except Exception:
                continue
        if not left and not right:
            return 0.0
        shared = min(len(left), len(right))
        distance = 0.0
        for index in range(shared):
            lval = left[index]
            rval = right[index]
            scale = max(abs(lval), abs(rval), 1.0)
            distance += abs(lval - rval) / scale
        distance += abs(len(left) - len(right))
        return distance

    @staticmethod
    def _profile_vector_distance(lhs: list[Any], rhs: list[Any]) -> float:
        left = []
        right = []
        for item in lhs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    left.append((float(item[0]), float(item[1])))
                except Exception:
                    continue
        for item in rhs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    right.append((float(item[0]), float(item[1])))
                except Exception:
                    continue
        if not left and not right:
            return 0.0
        shared = min(len(left), len(right))
        distance = 0.0
        for index in range(shared):
            l_proc, l_work = left[index]
            r_proc, r_work = right[index]
            proc_scale = max(abs(l_proc), abs(r_proc), 1.0)
            work_scale = max(abs(l_work), abs(r_work), 1.0)
            distance += abs(l_proc - r_proc) / proc_scale
            distance += abs(l_work - r_work) / work_scale
        distance += abs(len(left) - len(right)) * 2.0
        return distance

    def _rank_dispatch_cases(
        self,
        rows: list[tuple[Any, ...]],
        state: dict[str, Any],
        candidates: list[str],
    ) -> list[tuple[tuple[Any, ...], float, float]]:
        state_instance = self._state_instance_features(state)
        state_decision = self._state_decision_features(state, candidates)
        state_machine_type = str(state.get("machine_type", "") or "").strip()
        state_candidate_count = int(state_decision.get("candidate_count", 0) or 0)

        finite_makespans: list[float] = []
        for row in rows:
            metadata = self._safe_json_loads(str(row[3])) if len(row) > 3 else {}
            try:
                makespan_value = float(metadata.get("makespan", float("inf")))
            except Exception:
                makespan_value = float("inf")
            if makespan_value != float("inf"):
                finite_makespans.append(makespan_value)
        best_makespan = min(finite_makespans) if finite_makespans else None
        worst_makespan = max(finite_makespans) if finite_makespans else None

        ranked: list[tuple[tuple[Any, ...], float, float]] = []
        for row in rows:
            instance_features = self._safe_json_loads(str(row[4])) if len(row) > 4 else {}
            local_features = self._safe_json_loads(str(row[5])) if len(row) > 5 else {}
            state_summary = self._safe_json_loads(str(row[6])) if len(row) > 6 else {}
            metadata = self._safe_json_loads(str(row[3])) if len(row) > 3 else {}
            if isinstance(state_summary.get("candidate_jobs"), list):
                derived_features = self._candidate_set_features(
                    state_summary.get("candidate_jobs", []),
                    state_summary.get("default_action") or metadata.get("default_action"),
                )
                for key, value in derived_features.items():
                    local_features.setdefault(key, value)
            machine_type = str(row[1]) if len(row) > 1 else ""
            problem_scale_distance = self._feature_distance(
                state_instance,
                instance_features,
                ["n_jobs", "n_machines"],
            )
            instance_context_distance = self._feature_distance(
                state_instance,
                instance_features,
                ["remaining_jobs", "avg_job_work", "bottleneck_utilization"],
            )
            decision_distance = self._feature_distance(
                state_decision,
                local_features,
                [
                    "candidate_count",
                    "avg_proc_time",
                    "avg_remaining_work",
                    "avg_delta_makespan",
                    "min_proc_time",
                    "max_proc_time",
                    "min_remaining_work",
                    "max_remaining_work",
                    "min_delta_makespan",
                    "max_delta_makespan",
                    "bottleneck_queue",
                    "critical_candidate_ratio",
                    "proc_gap_12",
                    "rem_gap_12",
                    "delta_gap_12",
                    "default_proc_time",
                    "default_remaining_work",
                    "default_delta_makespan",
                    "default_criticality",
                    "default_queue_after",
                    "default_next_queue",
                    "global_remaining_jobs",
                    "global_avg_remaining_work",
                    "global_max_remaining_work",
                ],
            )
            vector_distance = self._profile_vector_distance(
                list(state_decision.get("candidate_profile_vector", [])),
                local_features.get("candidate_profile_vector", []),
            ) + self._vector_distance(
                list(state_decision.get("global_remaining_work_vector", [])),
                local_features.get("global_remaining_work_vector", []),
            )
            case_candidate_count = int(local_features.get("candidate_count", 0) or 0)
            candidate_count_gap = abs(state_candidate_count - case_candidate_count)
            machine_penalty = 0.0 if not state_machine_type or machine_type == state_machine_type else 2.5
            try:
                makespan = float(metadata.get("makespan", float("inf")))
            except Exception:
                makespan = float("inf")
            total_score = (
                (problem_scale_distance * 0.2)
                + (instance_context_distance * 0.6)
                + (decision_distance * 1.5)
                + (vector_distance * 1.8)
                + (candidate_count_gap * 0.6)
                + machine_penalty
            )
            ranked.append((row, total_score, makespan))

        ranked.sort(key=lambda item: (item[1], item[2]))
        return ranked[: self.rag_top_k]

    def _prefilter_dispatch_cases(
        self,
        rows: list[tuple[Any, ...]],
        state: dict[str, Any],
        candidates: list[str],
    ) -> list[tuple[Any, ...]]:
        if not rows:
            return rows
        state_decision = self._state_decision_features(state, candidates)
        state_candidate_count = int(state_decision.get("candidate_count", 0) or 0)
        filtered: list[tuple[Any, ...]] = []
        for row in rows:
            local_features = self._safe_json_loads(str(row[5])) if len(row) > 5 else {}
            case_candidate_count = int(local_features.get("candidate_count", 0) or 0)
            if abs(state_candidate_count - case_candidate_count) <= self.rag_prefilter_candidate_gap:
                filtered.append(row)
        return filtered if filtered else rows

    @staticmethod
    def _sort_numeric(values: list[float]) -> list[float]:
        return sorted((float(v) for v in values if v is not None), key=lambda x: x)

    def _should_use_rag_for_dispatch(self, state: dict[str, Any], candidates: list[str]) -> bool:
        if not self.use_rag:
            return False
        if not self.rag_conditional:
            return True
        if len(candidates) <= 1:
            return False
        candidate_jobs = state.get("candidate_jobs", []) if isinstance(state.get("candidate_jobs"), list) else []
        deltas = self._sort_numeric(
            [row.get("delta_makespan") for row in candidate_jobs if isinstance(row, dict)]
        )
        proc_times = self._sort_numeric(
            [row.get("proc_time_on_machine") for row in candidate_jobs if isinstance(row, dict)]
        )
        bottlenecks = state.get("bottleneck_machines", []) if isinstance(state.get("bottleneck_machines"), list) else []
        bottleneck_queue = 0.0
        if bottlenecks:
            try:
                bottleneck_queue = max(float(row.get("queue_size", 0.0) or 0.0) for row in bottlenecks if isinstance(row, dict))
            except Exception:
                bottleneck_queue = 0.0
        ambiguous_delta = False
        ambiguous_proc = False
        if len(deltas) >= 2:
            scale = max(abs(deltas[0]), abs(deltas[1]), 1.0)
            ambiguous_delta = abs(deltas[1] - deltas[0]) / scale <= self.rag_ambiguity_threshold
        if len(proc_times) >= 2:
            scale = max(abs(proc_times[0]), abs(proc_times[1]), 1.0)
            ambiguous_proc = abs(proc_times[1] - proc_times[0]) / scale <= self.rag_ambiguity_threshold
        return ambiguous_delta or ambiguous_proc or bottleneck_queue >= self.rag_bottleneck_threshold

    def _summarize_dispatch_case(
        self,
        row: tuple[Any, ...],
        score: float,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str:
        instance_name = str(row[0]) if len(row) > 0 else ""
        machine_type = str(row[1]) if len(row) > 1 else ""
        metadata = self._safe_json_loads(str(row[3])) if len(row) > 3 else {}
        local_features = self._safe_json_loads(str(row[5])) if len(row) > 5 else {}
        state_summary = self._safe_json_loads(str(row[6])) if len(row) > 6 else {}
        outcome = self._safe_json_loads(str(row[7])) if len(row) > 7 else {}

        chosen_action = str(state_summary.get("chosen_action") or metadata.get("chosen_action") or "")
        default_action = str(state_summary.get("default_action") or metadata.get("default_action") or "")
        candidate_jobs = state_summary.get("candidate_jobs", [])
        bottleneck_queue = local_features.get("bottleneck_queue")

        chosen_proc_rank = local_features.get("chosen_rank_by_proc")
        chosen_rem_rank = local_features.get("chosen_rank_by_remaining_work")
        makespan = metadata.get("makespan", outcome.get("final_makespan"))
        candidate_details = []

        chosen_profile = None
        default_profile = None
        if isinstance(candidate_jobs, list):
            for item in candidate_jobs:
                if not isinstance(item, dict):
                    continue
                profile = {
                    "job_id": item.get("job_id"),
                    "proc_time": item.get("proc_time_on_machine"),
                    "remaining_work": item.get("remaining_work"),
                    "remaining_ops": item.get("remaining_ops"),
                    "delta_makespan": item.get("delta_makespan"),
                    "queue_load_after": item.get("affected_machine_load_after"),
                    "critical": item.get("criticality_flag"),
                    "post_route": item.get("post_route"),
                }
                candidate_details.append(profile)
                if str(item.get("job_id")) == chosen_action:
                    chosen_profile = profile
                if str(item.get("job_id")) == default_action:
                    default_profile = profile

        other_profiles = [
            item for item in candidate_details
            if item != chosen_profile and item != default_profile
        ]
        proc_values = [
            float(item.get("proc_time"))
            for item in other_profiles
            if item.get("proc_time") is not None
        ]
        rem_values = [
            float(item.get("remaining_work"))
            for item in other_profiles
            if item.get("remaining_work") is not None
        ]
        other_range = {
            "proc": [min(proc_values), max(proc_values)] if proc_values else [],
            "rem_work": [min(rem_values), max(rem_values)] if rem_values else [],
            "count": len(other_profiles),
        }
        global_vec = local_features.get("global_remaining_work_vector", [])
        global_top3 = global_vec[:3] if isinstance(global_vec, list) else []

        hint_parts = []
        if chosen_proc_rank is not None:
            hint_parts.append(f"proc_rank={chosen_proc_rank}")
        if chosen_rem_rank is not None:
            hint_parts.append(f"rem_work_rank={chosen_rem_rank}")
        if bottleneck_queue is not None:
            hint_parts.append(f"bottleneck_queue={bottleneck_queue}")
        if global_top3:
            hint_parts.append(f"top_rem_work={global_top3}")

        preferred_profile = (
            f"proc={chosen_profile.get('proc_time')}, rem_work={chosen_profile.get('remaining_work')}, rem_ops={chosen_profile.get('remaining_ops')}"
            if isinstance(chosen_profile, dict)
            else "unavailable"
        )
        future_effect_parts: list[str] = []
        if isinstance(chosen_profile, dict):
            if chosen_profile.get("delta_makespan") is not None:
                future_effect_parts.append(f"delta_cmax={chosen_profile.get('delta_makespan')}")
            if chosen_profile.get("queue_load_after") is not None:
                future_effect_parts.append(f"queue_after={chosen_profile.get('queue_load_after')}")
            if chosen_profile.get("critical") is not None:
                future_effect_parts.append(f"critical={chosen_profile.get('critical')}")
            post_route = chosen_profile.get("post_route")
            if isinstance(post_route, dict):
                downstream_avg = post_route.get("downstream_avg_proc_sum")
                bottleneck_hits = post_route.get("downstream_bottleneck_hits")
                next_pressure = post_route.get("next_stage_queue_pressure")
                next_ready = post_route.get("next_stage_best_machine_ready")
                if downstream_avg is not None:
                    future_effect_parts.append(f"downstream_p={downstream_avg}")
                if bottleneck_hits is not None:
                    future_effect_parts.append(f"bneck_hits={bottleneck_hits}")
                if next_pressure is not None:
                    future_effect_parts.append(f"next_q={next_pressure}")
                if next_ready is not None:
                    future_effect_parts.append(f"next_ready={next_ready}")
        contrast_profile = (
            f"default={default_action}, other_candidates={other_range}"
            if default_action and default_action != chosen_action
            else f"other_candidates={other_range}"
        )

        if self.rag_explanatory_cases:
            return "\n".join(
                [
                    f"[DECISION_CASE] sim_score={score:.3f} instance={instance_name}",
                    f"- match: machine={machine_type}, candidates={len(candidate_jobs) if isinstance(candidate_jobs, list) else local_features.get('candidate_count')}",
                    f"- preferred_action={chosen_action or 'unknown'} ({contrast_profile})",
                    f"- preferred_profile: {preferred_profile}",
                    f"- future_effect_hint: {'; '.join(future_effect_parts) if future_effect_parts else 'unavailable'}",
                    f"- decision_hint: {'; '.join(hint_parts) if hint_parts else 'unavailable'}",
                    f"- outcome: final_makespan={makespan}",
                ]
            )

        return "\n".join(
            [
                f"[DECISION_CASE] sim_score={score:.3f} instance={instance_name}",
                f"- match: machine={machine_type}, candidates={len(candidate_jobs) if isinstance(candidate_jobs, list) else local_features.get('candidate_count')}",
                f"- preferred_action={chosen_action or 'unknown'} ({contrast_profile})",
                f"- preferred_profile: {preferred_profile}",
                f"- future_effect_hint: {'; '.join(future_effect_parts) if future_effect_parts else 'unavailable'}",
                f"- decision_hint: {'; '.join(hint_parts) if hint_parts else 'unavailable'}",
                f"- outcome: final_makespan={makespan}",
            ]
        )

    @staticmethod
    def _shorten_dispatch_memory(content: str) -> str:
        text = str(content).strip()
        if not text:
            return ""
        keep_keys = ("instance=", "time=", "machine_type=", "chosen_job=", "candidates=", "makespan=", "bottlenecks=")
        parts = [part.strip() for part in text.split(";")]
        kept = [part for part in parts if any(key in part for key in keep_keys)]
        return "; ".join(kept[:7]) if kept else text[:600]

    @staticmethod
    def _extract_chat_content(message: Any) -> str:
        if not isinstance(message, dict):
            return ""

        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
            return "\n".join(part for part in text_parts if part)

        refusal = message.get("refusal")
        if isinstance(refusal, str):
            return refusal
        return ""

    def _query_llm(self, prompt: str) -> str:
        return self._query_llm_with_limit(prompt, self.max_tokens)

    def _retry_backoff_sec(self, retry_count: int) -> float:
        return min(2 ** retry_count, 8.0)

    def _resolve_token_limit(self, state: dict[str, Any]) -> int:
        token_limit = int(self.max_tokens)
        if not self.adaptive_max_tokens:
            return token_limit
        problem_summary = state.get("problem_summary", {}) if isinstance(state.get("problem_summary"), dict) else {}
        total_jobs = int(problem_summary.get("total_jobs", 0) or 0)
        extra_buckets = max(0, (total_jobs - 1) // 10)
        token_limit += extra_buckets * int(self.adaptive_token_step)
        if self.use_rag:
            token_limit += max(50, int(self.adaptive_token_step // 2))
        if self.adaptive_token_cap is not None:
            token_limit = min(token_limit, int(self.adaptive_token_cap))
        return max(token_limit, int(self.max_tokens))

    def _query_llm_with_limit(self, prompt: str, token_limit: int, retry_count: int = 0) -> str:
        self.last_query_error = None
        self.last_response_body = None
        self.last_usage = None
        is_gpt5_family = str(self.model_name).lower().startswith("gpt-5")
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a job-shop scheduling agent. "
                        "Your objective is to minimize the final makespan. "
                        "Choose exactly one candidate action based on the current state and retrieved examples. "
                        "Prefer actions that reduce downstream congestion, avoid bottleneck idling, and lower remaining work on the critical path. "
                        "Return JSON only in the form {\"choice\": \"...\", \"reason\": \"...\"}."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        # Newer OpenAI chat models such as GPT-5 use max_completion_tokens.
        if is_gpt5_family:
            payload["max_completion_tokens"] = token_limit
        else:
            payload["temperature"] = self.temperature
            payload["max_tokens"] = token_limit
        try:
            payload_bytes = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            self.last_query_error = f"request_json_encode_error:{exc}"
            return ""
        req = urllib.request.Request(
            self.api_url,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
            self.last_response_body = body
            parsed = json.loads(body)
            usage = parsed.get("usage")
            self.last_usage = usage if isinstance(usage, dict) else None
            choices = parsed.get("choices")
            if not isinstance(choices, list) or not choices:
                self.last_query_error = "response_parse_error:missing_choices"
                return ""

            message = choices[0].get("message", {})
            content = self._extract_chat_content(message)
            if not content:
                finish_reason = choices[0].get("finish_reason")
                self.last_query_error = f"empty_model_content finish_reason={finish_reason!r}"
                # GPT-5 family may consume completion budget before emitting visible text.
                # Retry once with a larger completion-token budget.
                if (
                    is_gpt5_family
                    and finish_reason == "length"
                    and token_limit < max(self.max_tokens * 4, 1024)
                ):
                    retry_limit = max(token_limit * 2, 1024)
                    return self._query_llm_with_limit(prompt, retry_limit)
            return content
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            if exc.code in self.transient_retry_statuses and retry_count < self.max_transient_retries:
                time.sleep(self._retry_backoff_sec(retry_count))
                return self._query_llm_with_limit(prompt, token_limit, retry_count + 1)
            if (
                exc.code == 400
                and "not valid JSON" in error_body
                and retry_count < 1
            ):
                time.sleep(self._retry_backoff_sec(retry_count))
                return self._query_llm_with_limit(prompt, token_limit, retry_count + 1)
            self.last_query_error = f"http_error status={exc.code} body={error_body}"
            return ""
        except urllib.error.URLError as exc:
            if retry_count < self.max_transient_retries:
                time.sleep(self._retry_backoff_sec(retry_count))
                return self._query_llm_with_limit(prompt, token_limit, retry_count + 1)
            self.last_query_error = f"url_error reason={exc.reason}"
            return ""
        except (TimeoutError, socket.timeout) as exc:
            if retry_count < self.max_transient_retries:
                time.sleep(self._retry_backoff_sec(retry_count))
                return self._query_llm_with_limit(prompt, token_limit, retry_count + 1)
            self.last_query_error = f"timeout_error reason={exc}"
            return ""
        except (KeyError, IndexError, json.JSONDecodeError):
            self.last_query_error = "response_parse_error"
            return ""

    @staticmethod
    def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {
            "sim_time": state.get("sim_time"),
            "instance_name": state.get("instance_name"),
            "decision_type": state.get("decision_type"),
            "completed_jobs": state.get("completed_jobs"),
            "current_makespan": state.get("current_makespan"),
            "machine_ready_time": state.get("machine_ready_time"),
            "source_id": state.get("source_id"),
            "process_id": state.get("process_id"),
            "machine_type": state.get("machine_type"),
            "machine_id": state.get("machine_id"),
            "job_id": state.get("job_id"),
            "operation_id": state.get("operation_id"),
            "sequencing_rule": state.get("sequencing_rule"),
            "routing_rule": state.get("routing_rule"),
            "dispatching_rule": state.get("dispatching_rule"),
        }

        problem_summary = state.get("problem_summary")
        if isinstance(problem_summary, dict):
            compact["problem_summary"] = {
                "total_jobs": problem_summary.get("total_jobs"),
                "completed_jobs": problem_summary.get("completed_jobs"),
                "remaining_jobs": problem_summary.get("remaining_jobs"),
            }

        static_problem_context = state.get("static_problem_context")
        if isinstance(static_problem_context, dict):
            compact["static_problem_context"] = {
                "n_jobs": static_problem_context.get("n_jobs"),
                "n_machines": static_problem_context.get("n_machines"),
                "total_operations": static_problem_context.get("total_operations"),
                "avg_job_total_work": static_problem_context.get("avg_job_total_work"),
                "max_job_total_work": static_problem_context.get("max_job_total_work"),
            }

        global_dynamic_state = state.get("global_dynamic_state")
        if isinstance(global_dynamic_state, dict):
            machine_ready_times = global_dynamic_state.get("machine_ready_times")
            compact["global_dynamic_state"] = {
                "step_index": global_dynamic_state.get("step_index"),
                "current_makespan": global_dynamic_state.get("current_makespan"),
                "unfinished_jobs": global_dynamic_state.get("unfinished_jobs"),
                "total_remaining_work": global_dynamic_state.get("total_remaining_work"),
                "bottleneck_machine": global_dynamic_state.get("bottleneck_machine"),
                "machine_ready_times": machine_ready_times[:5] if isinstance(machine_ready_times, list) else [],
            }

        global_job_summary = state.get("global_job_summary")
        if isinstance(global_job_summary, list):
            compact["global_job_summary"] = [
                {
                    "job_id": row.get("job_id"),
                    "remaining_work": row.get("remaining_work"),
                    "remaining_ops": row.get("remaining_ops"),
                    "current_operation": row.get("current_operation"),
                    "status": row.get("status"),
                }
                for row in global_job_summary[:5]
                if isinstance(row, dict)
            ]

        recent_decisions = state.get("recent_decisions")
        if isinstance(recent_decisions, list):
            compact["recent_decisions"] = [
                {
                    "decision_type": row.get("decision_type"),
                    "sim_time": row.get("sim_time"),
                    "machine_type": row.get("machine_type"),
                    "default_action": row.get("default_action"),
                    "chosen_action": row.get("chosen_action"),
                }
                for row in recent_decisions[-3:]
                if isinstance(row, dict)
            ]

        process_queues = state.get("process_queues")
        if isinstance(process_queues, dict):
            compact["process_queue_sizes"] = {
                key: len(value) if isinstance(value, list) else 0 for key, value in process_queues.items()
            }

        machine_status = state.get("machine_status")
        if isinstance(machine_status, list):
            compact["machine_status"] = [
                {
                    "machine_instance": row.get("machine_instance"),
                    "machine_type": row.get("machine_type"),
                    "working": row.get("working"),
                    "queue_size": len(row.get("queued_jobs", [])) if isinstance(row.get("queued_jobs"), list) else 0,
                }
                for row in machine_status[:5]
                if isinstance(row, dict)
            ]

        candidate_jobs = state.get("candidate_jobs")
        if isinstance(candidate_jobs, list):
            compact["candidate_jobs"] = [
                {
                    "job_id": row.get("job_id"),
                    "arrival_time": row.get("arrival_time"),
                    "remaining_ops": row.get("remaining_ops"),
                    "total_ops": row.get("total_ops"),
                    "remaining_work": row.get("remaining_work"),
                    "proc_time_on_machine": row.get("proc_time_on_machine"),
                    "first_op_avg_time": row.get("first_op_avg_time"),
                    "job_progress_ratio": row.get("job_progress_ratio"),
                    "job_ready_time": row.get("job_ready_time"),
                    "estimated_start": row.get("estimated_start"),
                    "estimated_end": row.get("estimated_end"),
                    "post_action_makespan": row.get("post_action_makespan"),
                    "delta_makespan": row.get("delta_makespan"),
                    "delta_makespan_ratio": row.get("delta_makespan_ratio"),
                    "job_wait": row.get("job_wait"),
                    "machine_idle_gap": row.get("machine_idle_gap"),
                    "affected_machine_load": row.get("affected_machine_load"),
                    "affected_machine_ops_left": row.get("affected_machine_ops_left"),
                    "affected_machine_load_after": row.get("affected_machine_load_after"),
                    "affected_machine_ops_left_after": row.get("affected_machine_ops_left_after"),
                    "affected_machine_load_ratio": row.get("affected_machine_load_ratio"),
                    "remaining_work_after_ratio": row.get("remaining_work_after_ratio"),
                    "remaining_work_after_abs": row.get("remaining_work_after_abs"),
                    "slack_to_current_makespan": row.get("slack_to_current_makespan"),
                    "criticality_flag": row.get("criticality_flag"),
                    "post_route": row.get("post_route"),
                }
                for row in candidate_jobs[:10]
                if isinstance(row, dict)
            ]

        dispatch_candidates = state.get("dispatch_candidates")
        if isinstance(dispatch_candidates, list):
            compact["dispatch_candidates"] = dispatch_candidates[:10]

        routing_candidates = state.get("routing_candidates")
        if isinstance(routing_candidates, list):
            compact["routing_candidates"] = routing_candidates[:10]

        candidate_machines = state.get("candidate_machines")
        if isinstance(candidate_machines, list):
            compact["candidate_machines"] = [
                {
                    "machine_type": row.get("machine_type"),
                    "processing_time": row.get("processing_time"),
                    "queue_size": row.get("queue_size"),
                    "busy_instances": row.get("busy_instances"),
                }
                for row in candidate_machines[:10]
                if isinstance(row, dict)
            ]

        machine_status = compact.get("machine_status", [])
        if isinstance(machine_status, list) and machine_status:
            compact["bottleneck_machines"] = sorted(
                [
                    {
                        "machine_type": row.get("machine_type"),
                        "machine_instance": row.get("machine_instance"),
                        "working": row.get("working"),
                        "queue_size": row.get("queue_size"),
                    }
                    for row in machine_status
                ],
                key=lambda row: (row.get("queue_size", 0), 1 if row.get("working") else 0),
                reverse=True,
            )[:3]

        return {key: value for key, value in compact.items() if value not in (None, "", [], {})}

    @staticmethod
    def _objective_guidance(decision_type: str) -> str:
        if decision_type == "dispatch":
            return (
                "Objective: minimize final makespan. "
                "For dispatch, prefer the job that is most likely to reduce bottleneck waiting and shorten the critical path. "
                "Prioritize proc_time_on_machine, remaining_work, and machine congestion. "
                "Use remaining_ops only as a weak tie-breaker."
            )
        if decision_type == "routing":
            return (
                "Objective: minimize final makespan. "
                "For routing, prefer the machine choice that balances short processing time with low queue congestion and fewer busy instances."
            )
        if decision_type == "sequencing":
            return (
                "Objective: minimize final makespan. "
                "For sequencing, prefer the job release order that is likely to keep bottleneck machines utilized early and reduce total remaining work."
            )
        return "Objective: minimize final makespan."

    def _build_prompt(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
        rag: str,
    ) -> str:
        compact_state = self._compact_state(state)
        example_choice = candidates[0] if candidates else ""
        if self.response_mode == "choice_only":
            example_json = json.dumps({"choice": example_choice}, ensure_ascii=True)
        elif self.response_mode == "short_reason":
            example_json = json.dumps({"choice": example_choice, "reason": "very short"}, ensure_ascii=True)
        else:
            example_json = json.dumps({"choice": example_choice, "reason": "short explanation"}, ensure_ascii=True)
        if decision_type == "dispatch":
            candidate_jobs = compact_state.get("candidate_jobs", [])
            candidate_lines = []

            def _fmt_num(value: Any, digits: int = 1) -> str:
                try:
                    number = float(value)
                except Exception:
                    return "NA"
                if abs(number - round(number)) < 1e-9:
                    return str(int(round(number)))
                return f"{number:.{digits}f}"

            def _fmt_pct(value: Any) -> str:
                try:
                    number = float(value)
                except Exception:
                    return "NA"
                return f"{number * 100:.1f}%"

            def _fmt_post_route(value: Any) -> str:
                if not isinstance(value, dict):
                    return "NA"
                machine_route = value.get("remaining_machine_route")
                route_len = value.get("route_length_left")
                downstream_avg = value.get("downstream_avg_proc_sum")
                bottleneck_hits = value.get("downstream_bottleneck_hits")
                next_pressure = value.get("next_stage_queue_pressure")
                next_ready = value.get("next_stage_best_machine_ready")
                route_text = "->".join(str(item) for item in machine_route[:6]) if isinstance(machine_route, list) else "NA"
                if isinstance(machine_route, list) and len(machine_route) > 6:
                    route_text += "->..."
                return (
                    f"{route_text}, len={_fmt_num(route_len)}, "
                    f"downstream_p={_fmt_num(downstream_avg)}, "
                    f"bneck_hits={_fmt_num(bottleneck_hits)}, "
                    f"next_q={_fmt_num(next_pressure)}, "
                    f"next_ready={_fmt_num(next_ready)}"
                )

            for row in candidate_jobs:
                candidate_lines.append(
                    f"{row.get('job_id')}: "
                    f"p={_fmt_num(row.get('proc_time_on_machine'))}, "
                    f"rem_work={_fmt_num(row.get('remaining_work'))}, "
                    f"rem_ops={_fmt_num(row.get('remaining_ops'))}, "
                    f"progress={_fmt_pct(row.get('job_progress_ratio'))}, "
                    f"job_wait={_fmt_num(row.get('job_wait'))}, "
                    f"start_end={_fmt_num(row.get('estimated_start'))}->{_fmt_num(row.get('estimated_end'))}, "
                    f"delta_cmax={_fmt_num(row.get('delta_makespan'))}, "
                    f"slack={_fmt_num(row.get('slack_to_current_makespan'))}, "
                    f"machine_idle={_fmt_num(row.get('machine_idle_gap'))}, "
                    f"queue_load_after={_fmt_num(row.get('affected_machine_load_after'))}, "
                    f"queue_ops_after={_fmt_num(row.get('affected_machine_ops_left_after'))}, "
                    f"rem_work_after_abs={_fmt_num(row.get('remaining_work_after_abs'))}, "
                    f"rem_work_after={_fmt_pct(row.get('remaining_work_after_ratio'))}, "
                    f"critical={_fmt_num(row.get('criticality_flag'))}, "
                    f"route={_fmt_post_route(row.get('post_route'))}"
                )
            machine_focus = compact_state.get("bottleneck_machines", [])
            focus_line = json.dumps(machine_focus, ensure_ascii=True)
            problem_summary = compact_state.get("problem_summary", {})
            static_context = compact_state.get("static_problem_context", {})
            dynamic_context = compact_state.get("global_dynamic_state", {})
            global_summary = compact_state.get("global_job_summary", [])
            global_top = [
                row.get("remaining_work")
                for row in global_summary
                if isinstance(row, dict) and row.get("remaining_work") is not None
            ]
            global_avg = round(sum(global_top) / len(global_top), 1) if global_top else None
            near_finish = sum(
                1
                for row in global_summary
                if isinstance(row, dict) and isinstance(row.get("remaining_ops"), (int, float)) and row.get("remaining_ops") <= 1
            )
            global_line = json.dumps(
                {
                    "top_rem_work": global_top[:5],
                    "avg_rem_work_top": global_avg,
                    "near_finish_jobs": near_finish,
                },
                ensure_ascii=True,
            )
            recent_decisions = compact_state.get("recent_decisions", [])
            recent_line = " | ".join(
                f"t={row.get('sim_time')} {row.get('machine_type')} d={row.get('default_action')} c={row.get('chosen_action')}"
                for row in recent_decisions
            )
            prompt_lines = [
                "Objective: minimize final makespan for this dispatch decision.",
                f"Static problem context: {json.dumps(static_context, ensure_ascii=True)}",
                f"Global dynamic state: {json.dumps(dynamic_context, ensure_ascii=True)}",
                f"Problem summary: {json.dumps(problem_summary, ensure_ascii=True)}",
                f"Global remaining-work summary: {global_line}",
                f"Recent decisions: {recent_line}" if recent_line else "Recent decisions: none",
                f"Machine: {compact_state.get('machine_type')} at time {compact_state.get('sim_time')}",
                f"Current makespan estimate: {compact_state.get('current_makespan')}",
                f"Machine ready time: {compact_state.get('machine_ready_time')}",
                f"Candidates: {', '.join(candidates)}",
                f"Default action: {default_action}",
                f"Candidate stats: {' | '.join(candidate_lines)}",
                f"Bottleneck summary: {focus_line}",
                "Choose one job id.",
                "Prefer candidates with lower estimated makespan increase, less machine idle loss, and shorter critical-path work.",
                "Use proc_time_on_machine, remaining_work, delta_makespan, and post-route congestion together rather than any single field.",
                "Use remaining_ops only as a tie-breaker, not the main criterion.",
                "Do not repeatedly defer the largest remaining-work job on the critical path without strong evidence.",
                "If recent decisions have already deferred the same large remaining-work job multiple times, require a strong bottleneck or processing-time reason before deferring it again.",
            ]
            if self.response_mode == "choice_only":
                prompt_lines.extend(
                    [
                        "Return only JSON with exactly one key: choice.",
                        f"Example: {example_json}",
                    ]
                )
            elif self.response_mode == "short_reason":
                prompt_lines.extend(
                    [
                        "Return only JSON with keys choice and reason.",
                        "Keep the reason extremely short, at most 8 words.",
                        f"Example: {example_json}",
                    ]
                )
            else:
                prompt_lines.append(f"Return only JSON. Example: {example_json}")
            if rag:
                prompt_lines.append(
                    "Use retrieved cases only as supporting analogies. "
                    "Prioritize the current machine state, candidate stats, and bottleneck summary when a retrieved case is only partially similar."
                )
                prompt_lines.append(
                    "Do not copy a retrieved action unless its candidate structure and workload pattern clearly match the current state."
                )
                prompt_lines.append(f"Retrieved best prior cases:\n{rag}")
            return "\n".join(prompt_lines)
        prompt_lines = [
            self._objective_guidance(decision_type),
            f"Decision type: {decision_type}",
            f"Candidates: {', '.join(candidates)}",
            f"Default action: {default_action}",
            f"State: {json.dumps(compact_state, ensure_ascii=True)}",
            f'The choice value must be exactly one of: {", ".join(candidates)}',
        ]
        if self.response_mode == "choice_only":
            prompt_lines.extend(
                [
                    "Return exactly one JSON object with only the key choice.",
                    f"Valid example: {example_json}",
                    "Do not output a reason.",
                    "Do not explain the format. Do not output markdown. Output JSON only.",
                ]
            )
        elif self.response_mode == "short_reason":
            prompt_lines.extend(
                [
                    "Return exactly one JSON object with keys choice and reason.",
                    f"Valid example: {example_json}",
                    "The reason must be very short, at most 8 words.",
                    "Do not explain the format. Do not output markdown. Output JSON only.",
                ]
            )
        else:
            prompt_lines.extend(
                [
                    "Return exactly one JSON object with keys choice and reason.",
                    f"Valid example: {example_json}",
                    "The reason should briefly reference makespan, bottleneck usage, queue congestion, or remaining work.",
                    "Do not explain the format. Do not output markdown. Output JSON only.",
                ]
            )
        if rag:
            prompt_lines.append(f"Retrieved context:\n{rag}")
        return "\n".join(prompt_lines)

    def _log_prompt(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
        prompt: str,
    ) -> None:
        if self.show_prompt:
            print(f"\n=== LLM Prompt: {decision_type} ===\n{prompt}\n=== End Prompt ===\n")
        if self.prompt_logger is not None:
            self.prompt_logger.log(
                {
                    "decision_type": decision_type,
                    "state": self._compact_state(state),
                    "candidates": candidates,
                    "default_action": default_action,
                    "prompt": prompt,
                }
            )

    def _show_decision_result(
        self,
        decision_type: str,
        raw_response: str,
        parsed_choice: str | None,
        reason: str | None,
        error: str | None,
    ) -> None:
        if isinstance(self.last_usage, dict):
            print(
                "[Token Usage]"
                f" decision_type={decision_type}"
                f" prompt_tokens={self.last_usage.get('prompt_tokens')}"
                f" completion_tokens={self.last_usage.get('completion_tokens')}"
                f" total_tokens={self.last_usage.get('total_tokens')}"
            )
        if not self.show_prompt:
            return
        print(f"=== LLM Response: {decision_type} ===")
        print(raw_response)
        print("=== Parsed Decision ===")
        print(
            json.dumps(
                {
                    "choice": parsed_choice,
                    "reason": reason,
                    "error": error,
                },
                ensure_ascii=False,
            )
        )
        print("=== End Decision ===\n")

    def _resolve_trivial_action(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str | None:
        if not candidates:
            return default_action
        if len(candidates) != 1:
            return None

        choice = candidates[0]
        self.last_query_error = None
        self.last_response_body = None
        self.last_usage = None
        if self.decision_logger is not None:
            self.decision_logger.log(
                {
                    "decision_type": decision_type,
                    "state": state,
                    "candidates": candidates,
                    "default_action": default_action,
                    "raw_response": "",
                    "parsed_choice": choice,
                    "reason": "single_candidate_shortcut",
                    "error": None,
                    "query_error": None,
                    "response_body": None,
                    "usage": None,
                }
            )
        return choice

    def _retrieve_rag_context(
        self,
        state: dict[str, Any],
        candidates: list[str] | None = None,
        default_action: str | None = None,
    ) -> str:
        if not self.use_rag or not self.rag_db_path:
            return ""
        try:
            conn = sqlite3.connect(self.rag_db_path)
            decision_type = str(state.get("decision_type", "")).strip()
            machine_type = str(state.get("machine_type", "")).strip()
            if decision_type == "dispatch" and not self._should_use_rag_for_dispatch(state, candidates or []):
                conn.close()
                return ""
            query_text = " ".join(
                str(x)
                for x in [
                    decision_type,
                    state.get("job_id", ""),
                    state.get("operation_id", ""),
                    state.get("machine_type", ""),
                    state.get("process_id", ""),
                ]
                if x
            )

            if decision_type == "dispatch":
                if machine_type:
                    case_rows = conn.execute(
                        """
                        SELECT
                            c.instance_name,
                            c.machine_type,
                            c.content,
                            c.metadata_json,
                            i.feature_json,
                            c.local_features_json,
                            c.state_summary_json,
                            c.outcome_json
                        FROM decision_cases c
                        LEFT JOIN instance_features i ON i.run_id = c.run_id
                        WHERE c.decision_type = 'dispatch'
                          AND c.teacher_tier = 'teacher'
                          AND c.machine_type = ?
                        ORDER BY c.case_id DESC
                        LIMIT ?
                        """,
                        (machine_type, max(self.rag_top_k * 40, 200)),
                    ).fetchall()
                else:
                    case_rows = conn.execute(
                        """
                        SELECT
                            c.instance_name,
                            c.machine_type,
                            c.content,
                            c.metadata_json,
                            i.feature_json,
                            c.local_features_json,
                            c.state_summary_json,
                            c.outcome_json
                        FROM decision_cases c
                        LEFT JOIN instance_features i ON i.run_id = c.run_id
                        WHERE c.decision_type = 'dispatch'
                          AND c.teacher_tier = 'teacher'
                        ORDER BY c.case_id DESC
                        LIMIT ?
                        """,
                        (max(self.rag_top_k * 40, 200),),
                    ).fetchall()

                if not case_rows:
                    if machine_type:
                        case_rows = conn.execute(
                            """
                            SELECT
                                c.instance_name,
                                c.machine_type,
                                c.content,
                                c.metadata_json,
                                i.feature_json,
                                c.local_features_json,
                                c.state_summary_json,
                                c.outcome_json
                            FROM decision_cases c
                            LEFT JOIN instance_features i ON i.run_id = c.run_id
                            WHERE c.decision_type = 'dispatch'
                              AND c.machine_type = ?
                            ORDER BY c.case_id DESC
                            LIMIT ?
                            """,
                            (machine_type, max(self.rag_top_k * 40, 200)),
                        ).fetchall()
                    else:
                        case_rows = conn.execute(
                            """
                            SELECT
                                c.instance_name,
                                c.machine_type,
                                c.content,
                                c.metadata_json,
                                i.feature_json,
                                c.local_features_json,
                                c.state_summary_json,
                                c.outcome_json
                            FROM decision_cases c
                            LEFT JOIN instance_features i ON i.run_id = c.run_id
                            WHERE c.decision_type = 'dispatch'
                            ORDER BY c.case_id DESC
                            LIMIT ?
                            """,
                            (max(self.rag_top_k * 40, 200),),
                        ).fetchall()

                if case_rows:
                    case_rows = self._prefilter_dispatch_cases(case_rows, state, candidates or [])
                    ranked_cases = self._rank_dispatch_cases(case_rows, state, candidates or [])
                    conn.close()
                    return "\n".join(
                        self._summarize_dispatch_case(row, score, state, candidates or [], default_action)
                        for row, score, _makespan in ranked_cases
                    )

            memory_rows = []
            try:
                memory_rows = conn.execute(
                    """
                    SELECT m.instance_name, m.decision_type, m.content, m.metadata_json
                    FROM rag_decision_memories_fts f
                    JOIN rag_decision_memories m ON m.memory_id = f.rowid
                    WHERE rag_decision_memories_fts MATCH ?
                      AND m.decision_type = ?
                    LIMIT ?
                    """,
                    (query_text, decision_type, max(self.rag_top_k * 3, self.rag_top_k)),
                ).fetchall()
            except sqlite3.OperationalError:
                memory_rows = conn.execute(
                    """
                    SELECT instance_name, decision_type, content, metadata_json
                    FROM rag_decision_memories
                    WHERE decision_type = ?
                    ORDER BY memory_id DESC
                    LIMIT ?
                    """,
                    (decision_type, max(self.rag_top_k * 3, self.rag_top_k)),
                ).fetchall()

            if memory_rows:
                conn.close()
                ranked_rows = sorted(memory_rows, key=lambda row: self._memory_rank_key(row, ""))[: self.rag_top_k]
                if decision_type == "dispatch":
                    return "\n".join(
                        f"[RAG_MEMORY] instance={row[0]}, decision_type={row[1]}\n{self._shorten_dispatch_memory(row[2])}"
                        for row in ranked_rows
                    )
                return "\n".join(
                    f"[RAG_MEMORY] instance={row[0]}, decision_type={row[1]}\n{row[2]}"
                    for row in ranked_rows
                )

            chunk_rows = []
            try:
                chunk_rows = conn.execute(
                    """
                    SELECT c.instance_name, c.chunk_index, c.content
                    FROM rag_log_chunks_fts f
                    JOIN rag_log_chunks c ON c.chunk_id = f.rowid
                    WHERE rag_log_chunks_fts MATCH ?
                    LIMIT ?
                    """,
                    (query_text, self.rag_top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                chunk_rows = conn.execute(
                    """
                    SELECT instance_name, chunk_index, content
                    FROM rag_log_chunks
                    ORDER BY chunk_id DESC
                    LIMIT ?
                    """,
                    (self.rag_top_k,),
                ).fetchall()
            conn.close()
            if not chunk_rows:
                return ""
            return "\n".join(
                f"[RAG_CHUNK] instance={row[0]}, chunk={row[1]}\n{row[2]}"
                for row in chunk_rows
            )
        except Exception:
            return ""

    @staticmethod
    def _extract_json_text(raw: str) -> str | None:
        text = raw.strip()
        if not text:
            return None

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1].strip()
        return None

    @classmethod
    def _parse_response(
        cls,
        raw: str,
        candidates: list[str],
        response_mode: str = "verbose_reason",
    ) -> tuple[str | None, str | None, str | None]:
        if not raw:
            return None, None, "empty_response"

        text = cls._extract_json_text(raw)
        if text is None:
            return None, None, "invalid_json"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None, None, "invalid_json"

        choice = parsed.get("choice")
        reason = parsed.get("reason")
        if not isinstance(choice, str):
            return None, reason if isinstance(reason, str) else None, "missing_choice"
        if response_mode == "choice_only":
            reason = None
        if choice not in candidates:
            return choice, reason if isinstance(reason, str) else None, "choice_not_in_candidates"
        return choice, reason if isinstance(reason, str) else None, None

    def act(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str | None:
        trivial_choice = self._resolve_trivial_action(decision_type, state, candidates, default_action)
        if trivial_choice is not None:
            return trivial_choice
        rag = self._retrieve_rag_context(state, candidates, default_action)
        prompt = self._build_prompt(decision_type, state, candidates, default_action, rag)
        self._log_prompt(decision_type, state, candidates, default_action, prompt)
        raw = self._query_llm_with_limit(prompt, self._resolve_token_limit(state))
        choice, reason, error = self._parse_response(raw, candidates, self.response_mode)
        self._show_decision_result(decision_type, raw, choice, reason, error)
        if self.decision_logger is not None:
            self.decision_logger.log(
                {
                    "decision_type": decision_type,
                    "state": state,
                    "candidates": candidates,
                    "default_action": default_action,
                    "raw_response": raw,
                    "parsed_choice": choice,
                    "reason": reason,
                    "error": error,
                    "query_error": self.last_query_error,
                    "response_body": self.last_response_body,
                    "usage": self.last_usage,
                }
            )
        if error is not None and self.fail_on_invalid_action:
            raise AgentDecisionError(
                f"LLM agent returned invalid action for {decision_type}: {error}. "
                f"query_error={self.last_query_error!r} raw_response={raw!r}"
            )
        return choice if error is None else default_action


class HFSchedulingAgent(OpenAISchedulingAgent):
    """
    Hugging Face local model agent.
    Reuses the same compact RAG context and strict JSON parsing path as the API agent.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.0,
        max_new_tokens: int = 128,
        device: str = "auto",
        rag_db_path: str | None = None,
        rag_top_k: int = 2,
        use_rag: bool = True,
        decision_log_path: str | None = None,
        prompt_log_path: str | None = None,
        show_prompt: bool = False,
        fail_on_invalid_action: bool = True,
        response_mode: str = "verbose_reason",
    ):
        super().__init__(
            api_url="",
            api_key="",
            model=model_id,
            temperature=temperature,
            max_tokens=max_new_tokens,
            timeout_sec=0,
            rag_db_path=rag_db_path,
            rag_top_k=rag_top_k,
            use_rag=use_rag,
            decision_log_path=decision_log_path,
            prompt_log_path=prompt_log_path,
            show_prompt=show_prompt,
            fail_on_invalid_action=fail_on_invalid_action,
            response_mode=response_mode,
        )
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._generator = None
        self._tokenizer = None

    def _resolve_device(self) -> int:
        import torch

        if self.device == "cpu":
            return -1
        if self.device == "cuda":
            return 0
        return 0 if torch.cuda.is_available() else -1

    def _get_generator(self):
        if self._generator is not None:
            return self._generator

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self._resolve_device(),
        )
        return self._generator

    def _query_llm(self, prompt: str) -> str:
        generator = self._get_generator()
        system_text = (
            "You are a job-shop scheduling agent. "
            "Your objective is to minimize the final makespan. "
            "Output exactly one JSON object and nothing else. "
            "Use keys choice and reason. "
            "The choice value must exactly match one provided candidate. "
            "Prefer actions that reduce bottleneck waiting, queue congestion, and critical-path remaining work. "
            "Do not describe the format. Do not output markdown."
        )
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ]
            try:
                hf_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                hf_prompt = f"{system_text}\n\n{prompt}\n\nAnswer with JSON only:\n"
        else:
            hf_prompt = f"{system_text}\n\n{prompt}\n\nAnswer with JSON only:\n"
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "return_full_text": False,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        outputs = generator(hf_prompt, **generation_kwargs)
        if not outputs:
            return ""
        return outputs[0].get("generated_text", "").strip()


class ReinforcementLearningAgent(BaseSchedulingAgent):
    """
    Placeholder adapter for an RL policy model.
    Replace `policy_model.predict(...)` with the actual RL inference code.
    """

    def __init__(self, policy_model):
        self.policy_model = policy_model

    def act(
        self,
        decision_type: str,
        state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ) -> str | None:
        if not candidates:
            return default_action
        if self.policy_model is None:
            return default_action

        # Contract: RL model returns either an action id or an index into candidates.
        prediction = self.policy_model.predict(decision_type=decision_type, state=state, candidates=candidates)
        if isinstance(prediction, str) and prediction in candidates:
            return prediction
        if isinstance(prediction, int) and 0 <= prediction < len(candidates):
            return candidates[prediction]
        return default_action
