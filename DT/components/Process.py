import random as rd

import simpy

from .Job import Job


def _job_id_tiebreak_key(job: Job) -> tuple[int, str]:
    job_id = str(getattr(job, "id", ""))
    suffix = job_id[1:] if job_id.startswith("J") else job_id
    try:
        return (int(suffix), job_id)
    except ValueError:
        return (10**9, job_id)


class Process:
    """
    Represents a workstation/process. It selects a feasible machine and forwards jobs.
    """

    def __init__(
        self,
        model,
        resource,
        monitor,
        id,
        problem_data,
        env,
        dispatching_rule,
        routing_rule,
        decision_policy=None,
        decision_manager=None,
    ):
        self.model = model
        self.resource = resource
        self.monitor = monitor
        self.id = id
        self.proc_data = problem_data["machine_info"]
        self.env = env
        self.dispatching_rule = dispatching_rule
        self.routing_rule = routing_rule
        self.decision_policy = decision_policy
        self.decision_manager = decision_manager

        self.job_queue = simpy.FilterStore(self.env)
        self.env.process(self.run())

    def run(self):
        yield self.env.timeout(0)
        while True:
            job = yield self.job_queue.get()
            job.current_process = self
            job.last_process_entry_time = float(self.env.now)

            masking = {machine: False for machine in job.current_operation.machine_list}
            while True:
                if self.decision_manager is not None:
                    selected_machine = yield from self.route_job(job, masking)
                else:
                    selected_machine = self.routing(job, masking)

                if selected_machine is None:
                    for machine_type in job.current_operation.machine_list:
                        for machine in self.resource["Machine_pool"].machine_dict[machine_type]:
                            job.last_machine_queue_entry_time = float(self.env.now)
                            machine.job_queue.put(job)
                        job.status = "waiting"
                    break

                target_machine = next(
                    (
                        machine
                        for machine in self.resource["Machine_pool"].machine_dict[selected_machine]
                        if not machine.working
                    ),
                    None,
                )

                if target_machine is not None:
                    job.last_machine_queue_entry_time = float(self.env.now)
                    target_machine.job_queue.put(job)
                    target_machine.working = True
                    job.status = "working"
                    break

                masking[selected_machine] = True

    def to_next_process(self, job: Job, machine: str):
        job.complete_step()

        if not job.is_completed():
            next_operation = job.current_operation
            next_process_id = next_operation.process_list[0]
            job.last_operation_ready_time = float(self.env.now)

            print(f"{self.env.now:.2f}: Job {job.id} (Op: {next_operation.id}) -> Process {next_process_id}")

            yield self.model[next_process_id].job_queue.put(job)
            self.monitor.record(
                time=self.env.now,
                part_id=job.id,
                operation=next_operation.id,
                process=next_process_id,
                machine=None,
                event="Job Transferred",
            )
        else:
            job.completion_time = self.env.now
            yield self.model["Sink"].store.put(job)
            print(f"{self.env.now:.2f}: Job {job.id} completed -> Sink")
            self.monitor.record(
                time=self.env.now,
                part_id=job.id,
                operation=None,
                process="Sink",
                machine=None,
                event="Job Transferred to Sink",
            )

    def _dispatch(self) -> Job:
        queue = self.job_queue.items

        if self.dispatching_rule == "SPT":
            return min(
                queue,
                key=lambda j: (
                    j.current_operation.get_average_processing_time(),
                    _job_id_tiebreak_key(j),
                ),
            )
        if self.dispatching_rule == "WSPT":
            return min(
                queue,
                key=lambda j: j.current_operation.get_average_processing_time() / getattr(j, "weight", 1.0),
            )
        if self.dispatching_rule == "LPT":
            return max(queue, key=lambda j: j.current_operation.get_average_processing_time())
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

    def _default_routing_choice(self, job, masking=None) -> str | None:
        masking = masking or {}
        operation = job.current_operation
        available_machines = [m for m in operation.machine_list if not masking.get(m, False)]

        if len(available_machines) == 0:
            return None
        if len(available_machines) == 1:
            return available_machines[0]

        proc_times = operation.get_machine_process_time_map()
        if self.routing_rule == "SPT":
            return min(available_machines, key=lambda m: proc_times[m])
        if self.routing_rule == "WSPT":
            return min(available_machines, key=lambda m: proc_times[m] * getattr(job, "weight", 1.0))
        if self.routing_rule == "LPT":
            return max(available_machines, key=lambda m: proc_times[m])
        if self.routing_rule == "RANDOM":
            return rd.choice(available_machines)
        return available_machines[0]

    def route_job(self, job, masking=None):
        masking = masking or {}
        operation = job.current_operation
        available_machines = [m for m in operation.machine_list if not masking.get(m, False)]
        default_choice = self._default_routing_choice(job, masking)

        if len(available_machines) == 0:
            return None

        state = {
            "process_id": self.id,
            "job_id": job.id,
            "operation_id": operation.id,
            "routing_rule": self.routing_rule,
            "candidate_machines": [
                {
                    "machine_type": machine,
                    "processing_time": float(operation.get_machine_process_time_map()[machine]),
                    "queue_size": int(
                        sum(
                            len(inst.job_queue.items)
                            for inst in self.resource["Machine_pool"].machine_dict[machine]
                        )
                    ),
                    "busy_instances": int(
                        sum(
                            1
                            for inst in self.resource["Machine_pool"].machine_dict[machine]
                            if inst.working
                        )
                    ),
                }
                for machine in available_machines
            ],
        }
        selected = yield self.decision_manager.submit(
            decision_type="routing",
            local_state=state,
            candidates=available_machines,
            default_action=default_choice,
        )
        return selected if selected in available_machines else default_choice

    def routing(self, job, masking=None) -> str | None:
        masking = masking or {}
        operation = job.current_operation
        available_machines = [m for m in operation.machine_list if not masking.get(m, False)]
        default_choice = self._default_routing_choice(job, masking)

        if self.decision_policy is None:
            return default_choice

        state = {
            "decision_type": "routing",
            "time": float(self.env.now),
            "process_id": self.id,
            "job_id": job.id,
            "operation_id": operation.id,
            "routing_rule": self.routing_rule,
            "candidate_machines": [
                {
                    "machine_type": machine,
                    "processing_time": float(operation.get_machine_process_time_map()[machine]),
                    "is_masked": bool(masking.get(machine, False)),
                }
                for machine in available_machines
            ],
        }
        chosen = self.decision_policy.choose_routing(
            state=state,
            candidate_machine_types=available_machines,
            default_choice=default_choice,
        )
        if chosen in available_machines:
            return chosen
        return default_choice
