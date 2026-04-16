import random as rd
from typing import Dict

import simpy

from DT.components import Job


def _job_id_tiebreak_key(job: Job) -> tuple[int, str]:
    job_id = str(getattr(job, "id", ""))
    suffix = job_id[1:] if job_id.startswith("J") else job_id
    try:
        return (int(suffix), job_id)
    except ValueError:
        return (10**9, job_id)


class Machine_pool:
    """
    Owns all machine instances and machine-type stores.
    """

    def __init__(
        self,
        monitor,
        problem_data: Dict,
        env: simpy.Environment,
        dispatching_rule: str,
        decision_policy=None,
        decision_manager=None,
    ):
        self.monitor = monitor
        self.env = env
        self.machine_data = problem_data["machine_info"]
        self.machine_dict = {}
        self.machine_list = []
        self.machine_stores = {}
        self.decision_policy = decision_policy
        self.decision_manager = decision_manager

        for machine_info in self.machine_data.values():
            self.machine_dict[machine_info["id"]] = []
            self.machine_stores[machine_info["id"]] = simpy.Store(env)
            for i in range(machine_info["capacity"]):
                machine = Machines(
                    machine_info["id"],
                    machine_info["id"] + f"_{i + 1}",
                    machine_info["processes"],
                    env,
                    dispatching_rule,
                    self,
                    decision_policy=decision_policy,
                    decision_manager=decision_manager,
                )
                self.machine_dict[machine_info["id"]].append(machine)
                self.machine_list.append(machine)
                self.env.process(machine.run(self))
                self.machine_stores[machine.type].put(machine)


class Machines:
    """
    Single machine instance.
    """

    def __init__(
        self,
        type: str,
        id: str,
        proc_data: list,
        env: simpy.Environment,
        dispatching_rule: str,
        machines,
        decision_policy=None,
        decision_manager=None,
    ):
        self.type = type
        self.id = id
        self.processes = proc_data
        self.env = env
        self.machines = machines
        self.working = False
        self.job_queue = simpy.FilterStore(self.env)
        self.dispatching_rule = dispatching_rule
        self.decision_policy = decision_policy
        self.decision_manager = decision_manager
        self.last_finish_time = 0.0
        self.last_start_time = -1.0
        self.current_job = None

    def run(self, machines):
        while True:
            yield self.machines.machine_stores[self.type].get()

            while any(job.status == "transporting" for job in self.machines.monitor.job_list):
                yield self.env.timeout(0)

            if len(self.job_queue.items) > 1:
                if self.decision_manager is not None:
                    selected_job = yield from self.dispatch_job()
                else:
                    selected_job = self._dispatch()
                job = yield self.job_queue.get(lambda item: item.id == selected_job.id)
            else:
                job = yield self.job_queue.get()

            print(f"{self.env.now:.2f}: Job {job.id} (Machine: {self.id}) assigned")
            self.last_start_time = float(self.env.now)
            job.last_dispatch_start_time = float(self.env.now)
            self.current_job = job

            for machine in machines.machine_list:
                if job in machine.job_queue.items:
                    machine.job_queue.items.remove(job)

            job.current_process.monitor.record(
                time=self.env.now,
                part_id=job.id,
                operation=job.current_operation.id,
                process=self.id,
                machine=self.type,
                event="Job Assigned",
            )

            self.env.process(self.processing(job))

    def processing(self, job: Job):
        operation = job.current_operation
        processing_time = operation.get_processing_time_for_machine(self.type)

        yield self.env.timeout(processing_time)
        job.status = "transporting"
        self.last_finish_time = float(self.env.now)
        self.current_job = None

        job.current_process.monitor.record(
            time=self.env.now,
            part_id=job.id,
            operation=operation.id,
            process=self.id,
            machine=self.type,
            event="Operation Complete",
        )

        self.env.process(job.current_process.to_next_process(job, self))
        yield self.machines.machine_stores[self.type].put(self)
        self.working = False

    def _default_dispatch_choice(self) -> Job:
        queue = self.job_queue.items
        if self.dispatching_rule == "SPT":
            return min(
                queue,
                key=lambda j: (
                    j.current_operation.get_processing_time_for_machine(self.type),
                    _job_id_tiebreak_key(j),
                ),
            )
        if self.dispatching_rule == "WSPT":
            return min(
                queue,
                key=lambda j: j.current_operation.get_processing_time_for_machine(self.type) * getattr(j, "weight", 1.0),
            )
        if self.dispatching_rule == "LPT":
            return max(queue, key=lambda j: j.current_operation.get_processing_time_for_machine(self.type))
        if self.dispatching_rule in ["MWKR", "LWKR"]:
            def remaining_work(job: Job):
                rem_time = job.current_operation.get_average_processing_time()
                rem_ops = job.operation_list[job.step + 1 :]
                if rem_ops:
                    rem_time += sum(op.get_average_processing_time() for op in rem_ops)
                return rem_time

            if self.dispatching_rule == "MWKR":
                return max(queue, key=remaining_work)
            return min(queue, key=remaining_work)
        if self.dispatching_rule == "RANDOM":
            return rd.choice(queue)
        return queue[0]

    def dispatch_job(self):
        queue = self.job_queue.items
        default_choice = self._default_dispatch_choice()
        current_time = float(self.env.now)
        all_jobs = list(getattr(self.machines.monitor, "job_list", []))

        def remaining_work(job: Job) -> float:
            rem_ops = job.operation_list[job.step :]
            return float(sum(op.get_average_processing_time() for op in rem_ops))

        def current_makespan_estimate() -> float:
            completed = [
                float(getattr(job, "completion_time", -1.0))
                for job in self.machines.monitor.job_list
                if getattr(job, "completion_time", -1.0) >= 0.0
            ]
            return float(max(max(completed) if completed else current_time, current_time))

        current_makespan = current_makespan_estimate()
        machine_ready_time = float(self.last_finish_time)
        machine_idle_gap = max(0.0, current_time - machine_ready_time)
        queued_machine_load = sum(
            float(job.current_operation.get_processing_time_for_machine(self.type))
            for job in queue
        )
        queued_machine_ops_left = sum(max(len(job.operation_list) - job.step, 0) for job in queue)
        global_remaining_work = sorted(
            [remaining_work(job) for job in all_jobs if not job.is_completed()],
            reverse=True,
        )
        critical_threshold = global_remaining_work[:3]

        state = {
            "machine_instance": self.id,
            "machine_type": self.type,
            "dispatch_rule": self.dispatching_rule,
            "current_makespan": current_makespan,
            "machine_ready_time": machine_ready_time,
            "candidate_jobs": [
                {
                    "job_id": job.id,
                    "operation_id": job.current_operation.id if job.current_operation else "",
                    "proc_time_on_machine": float(job.current_operation.get_processing_time_for_machine(self.type)),
                    "remaining_ops": len(job.operation_list) - job.step,
                    "remaining_work": remaining_work(job),
                    "total_ops": len(job.operation_list),
                    "job_progress_ratio": (
                        float(job.step) / float(len(job.operation_list))
                        if len(job.operation_list) > 0
                        else 0.0
                    ),
                    "job_ready_time": float(getattr(job, "last_machine_queue_entry_time", current_time)),
                    "estimated_start": current_time,
                    "estimated_end": current_time + float(job.current_operation.get_processing_time_for_machine(self.type)),
                    "post_action_makespan": max(
                        current_makespan,
                        current_time + float(job.current_operation.get_processing_time_for_machine(self.type)),
                    ),
                    "delta_makespan": max(
                        0.0,
                        max(
                            current_makespan,
                            current_time + float(job.current_operation.get_processing_time_for_machine(self.type)),
                        ) - current_makespan,
                    ),
                    "delta_makespan_ratio": (
                        max(
                            0.0,
                            max(
                                current_makespan,
                                current_time + float(job.current_operation.get_processing_time_for_machine(self.type)),
                            ) - current_makespan,
                        ) / current_makespan
                        if current_makespan > 0.0
                        else 0.0
                    ),
                    "job_wait": max(
                        0.0,
                        current_time - float(getattr(job, "last_machine_queue_entry_time", current_time)),
                    ),
                    "machine_idle_gap": machine_idle_gap,
                    "affected_machine_load": queued_machine_load,
                    "affected_machine_ops_left": queued_machine_ops_left,
                    "affected_machine_load_after": queued_machine_load
                    - float(job.current_operation.get_processing_time_for_machine(self.type)),
                    "affected_machine_ops_left_after": max(
                        0,
                        queued_machine_ops_left - max(len(job.operation_list) - job.step, 0),
                    ),
                    "affected_machine_load_ratio": (
                        float(job.current_operation.get_processing_time_for_machine(self.type)) / queued_machine_load
                        if queued_machine_load > 0.0
                        else 0.0
                    ),
                    "remaining_work_after_ratio": (
                        max(
                            0.0,
                            remaining_work(job)
                            - float(job.current_operation.get_processing_time_for_machine(self.type)),
                        )
                        / remaining_work(job)
                        if remaining_work(job) > 0.0
                        else 0.0
                    ),
                    "remaining_work_after_abs": max(
                        0.0,
                        remaining_work(job)
                        - float(job.current_operation.get_processing_time_for_machine(self.type)),
                    ),
                    "slack_to_current_makespan": current_makespan
                    - (
                        current_time + float(job.current_operation.get_processing_time_for_machine(self.type))
                    ),
                    "criticality_flag": 1.0 if remaining_work(job) in critical_threshold else 0.0,
                    "post_route": self._summarize_post_route(job),
                }
                for job in queue
            ],
        }
        chosen_job_id = yield self.decision_manager.submit(
            decision_type="dispatch",
            local_state=state,
            candidates=[job.id for job in queue],
            default_action=default_choice.id if default_choice is not None else None,
        )
        selected = next((job for job in queue if job.id == chosen_job_id), None)
        return selected if selected is not None else default_choice

    def _summarize_post_route(self, job: Job) -> dict:
        next_index = job.step + 1
        if next_index >= len(job.operation_list):
            return {
                "remaining_machine_route": [],
                "route_length_left": 0,
                "downstream_avg_proc_sum": 0.0,
                "downstream_bottleneck_hits": 0.0,
                "next_stage_queue_pressure": 0.0,
                "next_stage_best_machine_ready": 0.0,
            }

        remaining_ops = job.operation_list[next_index:]
        remaining_machine_route = []
        downstream_avg_proc_sum = 0.0
        downstream_bottleneck_hits = 0.0
        bottleneck_machine_types = {
            machine.type
            for machine in self.machines.machine_list
            if len(machine.job_queue.items) > 0 or getattr(machine, "working", False)
        }

        next_op = remaining_ops[0]
        machine_options = list(next_op.machine_list)
        next_stage_queue_pressure = 0.0
        next_stage_best_machine_ready = None

        for op in remaining_ops:
            downstream_avg_proc_sum += float(op.get_average_processing_time())
            chosen_machine = op.machine_list[0] if op.machine_list else None
            if chosen_machine is not None:
                remaining_machine_route.append(chosen_machine)
                if chosen_machine in bottleneck_machine_types:
                    downstream_bottleneck_hits += 1.0

        for machine_type in machine_options:
            machine_group = self.machines.machine_dict.get(machine_type, [])
            queue_pressure = sum(len(machine.job_queue.items) for machine in machine_group)
            next_stage_queue_pressure += float(queue_pressure)
            for machine in machine_group:
                candidate_ready = float(machine.last_finish_time) if getattr(machine, "working", False) else float(self.env.now)
                if next_stage_best_machine_ready is None or candidate_ready < next_stage_best_machine_ready:
                    next_stage_best_machine_ready = candidate_ready
        return {
            "remaining_machine_route": remaining_machine_route,
            "route_length_left": len(remaining_machine_route),
            "downstream_avg_proc_sum": downstream_avg_proc_sum,
            "downstream_bottleneck_hits": downstream_bottleneck_hits,
            "next_stage_queue_pressure": next_stage_queue_pressure,
            "next_stage_best_machine_ready": float(next_stage_best_machine_ready or self.env.now),
        }

    def _dispatch(self) -> Job:
        default_choice = self._default_dispatch_choice()
        if self.decision_policy is None:
            return default_choice

        queue = self.job_queue.items
        state = {
            "decision_type": "dispatch",
            "time": float(self.env.now),
            "machine_instance": self.id,
            "machine_type": self.type,
            "dispatch_rule": self.dispatching_rule,
            "candidate_jobs": [
                {
                    "job_id": job.id,
                    "operation_id": job.current_operation.id if job.current_operation else "",
                    "proc_time_on_machine": float(job.current_operation.get_processing_time_for_machine(self.type)),
                    "remaining_ops": len(job.operation_list) - job.step,
                }
                for job in queue
            ],
        }
        chosen_job_id = self.decision_policy.choose_dispatch(
            state=state,
            candidate_job_ids=[job.id for job in queue],
            default_choice=default_choice.id if default_choice is not None else None,
        )
        selected = next((job for job in queue if job.id == chosen_job_id), None)
        return selected if selected is not None else default_choice
