from collections import deque
from dataclasses import dataclass
from typing import Any

from simpy.core import EmptySchedule

from DT.agents import AgentDecisionError


@dataclass
class DecisionRequest:
    decision_type: str
    state: dict[str, Any]
    candidates: list[str]
    default_action: str | None
    response_event: Any


class DecisionManager:
    """
    Bridges SimPy processes and an external agent loop.
    Components submit decision requests and wait on `response_event`.
    The outer runner advances `env.step()`, pulls pending requests, asks the agent,
    and resolves the event with the chosen action.
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.monitor = None
        self.model = None
        self.resource = None
        self.problem_data = None
        self.pending_requests: deque[DecisionRequest] = deque()
        self.decision_count = 0
        self.recent_decisions: deque[dict[str, Any]] = deque(maxlen=6)

    def bind_runtime(self, env, monitor, model, resource) -> None:
        self.env = env
        self.monitor = monitor
        self.model = model
        self.resource = resource
        if hasattr(self.agent, "bind_runtime"):
            self.agent.bind_runtime(env=env, monitor=monitor, model=model, resource=resource)

    def _global_snapshot(self) -> dict[str, Any]:
        process_queues = {}
        if self.model is not None:
            for name, component in self.model.items():
                if hasattr(component, "job_queue"):
                    process_queues[name] = [job.id for job in component.job_queue.items]

        machine_status = []
        machine_ready_times = []
        machine_pool = None
        if self.resource is not None:
            machine_pool = self.resource.get("Machine_pool")
        if machine_pool is not None:
            for machine in machine_pool.machine_list:
                ready_time = float(getattr(machine, "last_finish_time", self.env.now))
                machine_status.append(
                    {
                        "machine_instance": machine.id,
                        "machine_type": machine.type,
                        "working": bool(machine.working),
                        "queued_jobs": [job.id for job in machine.job_queue.items],
                        "ready_time": ready_time,
                    }
                )
                machine_ready_times.append(
                    {
                        "machine_instance": machine.id,
                        "machine_type": machine.type,
                        "ready_time": ready_time,
                    }
                )

        sink_parts = 0
        sink = self.model.get("Sink") if self.model is not None else None
        if sink is not None and hasattr(sink, "parts_rec"):
            sink_parts = int(sink.parts_rec)

        problem_summary = {}
        global_jobs = []
        static_problem_context = {}
        global_dynamic_state = {}
        source = self.model.get("Source") if self.model is not None else None
        if source is not None and hasattr(source, "monitor"):
            all_jobs = list(getattr(source.monitor, "job_list", []))
            total_operations = sum(len(job.operation_list) for job in all_jobs)
            job_total_work = [
                float(sum(op.get_average_processing_time() for op in job.operation_list))
                for job in all_jobs
            ]
            problem_summary = {
                "total_jobs": len(all_jobs),
                "completed_jobs": sink_parts,
                "remaining_jobs": sum(1 for job in all_jobs if not job.is_completed()),
            }
            for job in all_jobs:
                if job.is_completed():
                    continue
                remaining_ops = max(len(job.operation_list) - job.step, 0)
                remaining_work = float(
                    sum(op.get_average_processing_time() for op in job.operation_list[job.step :])
                )
                global_jobs.append(
                    {
                        "job_id": job.id,
                        "remaining_ops": remaining_ops,
                        "remaining_work": remaining_work,
                        "current_operation": job.current_operation.id if job.current_operation else None,
                        "status": getattr(job, "status", None),
                    }
                )
            global_jobs.sort(key=lambda row: row["remaining_work"], reverse=True)
            static_problem_context = {
                "n_jobs": len(all_jobs),
                "n_machines": len(machine_status),
                "total_operations": total_operations,
                "avg_job_total_work": float(sum(job_total_work) / len(job_total_work)) if job_total_work else 0.0,
                "max_job_total_work": float(max(job_total_work)) if job_total_work else 0.0,
            }
            total_remaining_work = float(sum(row["remaining_work"] for row in global_jobs)) if global_jobs else 0.0
            bottleneck_machine = None
            if machine_status:
                ranked_bottlenecks = sorted(
                    machine_status,
                    key=lambda row: (
                        len(row.get("queued_jobs", [])) if isinstance(row.get("queued_jobs"), list) else 0,
                        1 if row.get("working") else 0,
                    ),
                    reverse=True,
                )
                if ranked_bottlenecks:
                    top = ranked_bottlenecks[0]
                    bottleneck_machine = {
                        "machine_type": top.get("machine_type"),
                        "machine_instance": top.get("machine_instance"),
                        "queue_size": len(top.get("queued_jobs", [])) if isinstance(top.get("queued_jobs"), list) else 0,
                        "ready_time": top.get("ready_time"),
                    }
            global_dynamic_state = {
                "step_index": int(self.decision_count + 1),
                "current_makespan": float(self.env.now),
                "unfinished_jobs": int(problem_summary.get("remaining_jobs", 0) or 0),
                "total_remaining_work": total_remaining_work,
                "machine_ready_times": machine_ready_times[:10],
                "bottleneck_machine": bottleneck_machine,
            }

        return {
            "sim_time": float(self.env.now),
            "instance_name": (self.problem_data or {}).get("instance_name"),
            "process_queues": process_queues,
            "machine_status": machine_status,
            "completed_jobs": sink_parts,
            "problem_summary": problem_summary,
            "static_problem_context": static_problem_context,
            "global_dynamic_state": global_dynamic_state,
            "global_job_summary": global_jobs[:10],
            "recent_decisions": list(self.recent_decisions),
        }

    def submit(
        self,
        decision_type: str,
        local_state: dict[str, Any],
        candidates: list[str],
        default_action: str | None,
    ):
        state = self._global_snapshot()
        state.update(local_state)
        state["decision_type"] = decision_type

        response_event = self.env.event()
        request = DecisionRequest(
            decision_type=decision_type,
            state=state,
            candidates=candidates,
            default_action=default_action,
            response_event=response_event,
        )
        self.pending_requests.append(request)
        return response_event

    def has_pending(self) -> bool:
        return bool(self.pending_requests)

    def pop_next(self) -> DecisionRequest:
        return self.pending_requests.popleft()

    def resolve(self, request: DecisionRequest, action: str | None) -> None:
        chosen = action if action in request.candidates else request.default_action
        self.decision_count += 1
        self.recent_decisions.append(
            {
                "decision_type": request.decision_type,
                "sim_time": float(request.state.get("sim_time", self.env.now)),
                "machine_type": request.state.get("machine_type"),
                "candidates": list(request.candidates),
                "default_action": request.default_action,
                "chosen_action": chosen,
            }
        )
        request.response_event.succeed(chosen)


def run_env_with_agent(env, decision_manager: DecisionManager):
    """
    Advance the SimPy environment one event at a time.
    Whenever a component blocks on a decision request, call the agent and resume.
    """

    while True:
        while decision_manager.has_pending():
            request = decision_manager.pop_next()
            try:
                action = decision_manager.agent.act(
                    decision_type=request.decision_type,
                    state=request.state,
                    candidates=request.candidates,
                    default_action=request.default_action,
                )
            except AgentDecisionError:
                raise
            decision_manager.resolve(request, action)

        try:
            env.step()
        except EmptySchedule:
            break
