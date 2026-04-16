import random as rd
from collections import defaultdict
from typing import List

from .Job import Job


class Source:
    """
    Creates jobs from input data and injects them into the first process.
    """

    def __init__(
        self,
        model,
        monitor,
        id,
        problem_data,
        env,
        sequencing_rule,
        routing_rule,
        decision_policy=None,
        decision_manager=None,
    ):
        self.model = model
        self.monitor = monitor
        self.id = id
        self.job_data = problem_data["job_info"]
        self.job_id_list = list(self.job_data.keys())
        self.operation_data = problem_data["operation_info"]
        self.env = env
        self.sequencing_rule = sequencing_rule
        self.routing_rule = routing_rule
        self.decision_policy = decision_policy
        self.decision_manager = decision_manager
        self._initial_batch_released = False
        self.env.process(self.job_generator())

    @staticmethod
    def _total_avg_time(job: Job) -> float:
        return float(sum(op.get_average_processing_time() for op in job.operation_list))

    def job_generator(self):
        batch_job_list = []
        sorted_job_ids = sorted(self.job_id_list, key=lambda jid: self.job_data[jid]["arrival_time"])

        for job_id in sorted_job_ids:
            arrival_time = self.job_data[job_id]["arrival_time"]
            if arrival_time > self.env.now:
                if batch_job_list:
                    yield from self.release_batch(batch_job_list)
                    batch_job_list = []
                yield self.env.timeout(arrival_time - self.env.now)

            job = Job(self.job_data[job_id], self.operation_data)
            job.last_operation_ready_time = float(self.env.now)
            job.last_process_entry_time = float(self.env.now)
            self.monitor.job_list.append(job)
            self.monitor.record(
                time=self.env.now,
                part_id=job.id,
                operation=None,
                process=self.id,
                machine=None,
                event="Job Created",
            )
            print(f"{self.env.now:.2f}: Job {job.id} created (arrival_time: {job.arrival_time}).")
            batch_job_list.append(job)

        if batch_job_list:
            yield from self.release_batch(batch_job_list)

    def release_batch(self, batch_job_list: List[Job]):
        # For the initial t=0 batch in fixed-machine JSSP, bypass source-level
        # sequencing and expose the first-operation candidates directly to the
        # corresponding machine queues so the existing machine dispatch rule
        # determines the initial assignment.
        if self._can_use_initial_machine_wise_release(batch_job_list):
            self._initial_batch_released = True
            self._release_initial_batch_machine_wise(batch_job_list)
            return

        adjusted_batch = yield from self.sequence_jobs(batch_job_list)
        for job in adjusted_batch:
            self.env.process(self.to_next_process(job))

    def _can_use_initial_machine_wise_release(self, batch_job_list: List[Job]) -> bool:
        if self._initial_batch_released or self.env.now != 0:
            return False
        if not batch_job_list:
            return False
        return all(len(job.current_operation.machine_list) == 1 for job in batch_job_list)

    def _release_initial_batch_machine_wise(self, batch_job_list: List[Job]):
        machine_groups = defaultdict(list)

        for job in batch_job_list:
            next_operation = job.current_operation
            next_process_id = next_operation.process_list[0]
            next_machine_type = next_operation.machine_list[0]
            next_process = self.model[next_process_id]

            job.current_process = next_process
            job.last_operation_ready_time = float(self.env.now)
            job.last_process_entry_time = float(self.env.now)
            job.last_machine_queue_entry_time = float(self.env.now)
            job.status = "waiting"

            print(f"{self.env.now:.2f}: Job {job.id} (Op: {next_operation.id}) first -> Process {next_process_id}")
            self.monitor.record(
                time=self.env.now,
                part_id=job.id,
                operation=next_operation.id,
                process=next_process_id,
                machine=None,
                event="Job Transferred",
            )
            machine_groups[next_machine_type].append(job)

        for machine_type, jobs in machine_groups.items():
            machine_group = self.model[jobs[0].current_process.id].resource["Machine_pool"].machine_dict[machine_type]
            for machine in machine_group:
                for job in jobs:
                    machine.job_queue.put(job)

    def _default_sequence(self, batch_job_list: List[Job]) -> List[Job]:
        def johnson(job_list: List[Job]) -> List[Job]:
            op_times = {
                job: [op.get_average_processing_time() for op in job.operation_list]
                for job in job_list
            }
            first_group, second_group = [], []
            for job in job_list:
                if op_times[job][0] < op_times[job][-1]:
                    first_group.append(job)
                else:
                    second_group.append(job)
            first_group.sort(key=lambda j: op_times[j][0])
            second_group.sort(key=lambda j: op_times[j][-1], reverse=True)
            return first_group + second_group

        def palmer(job_list: List[Job]) -> List[Job]:
            op_times = {
                job: [op.get_average_processing_time() for op in job.operation_list]
                for job in job_list
            }
            if not job_list:
                return []
            num_machines = len(op_times[job_list[0]])
            slope_indices = {}
            for job in job_list:
                index = sum(
                    (num_machines - (2 * (i + 1)) + 1) * proc_time
                    for i, proc_time in enumerate(op_times[job])
                )
                slope_indices[job] = index
            return sorted(job_list, key=lambda j: slope_indices[j], reverse=True)

        if self.sequencing_rule == "SPT":
            return sorted(batch_job_list, key=self._total_avg_time)
        if self.sequencing_rule == "LPT":
            return sorted(batch_job_list, key=self._total_avg_time, reverse=True)
        if self.sequencing_rule == "FSPT":
            return sorted(batch_job_list, key=lambda j: j.operation_list[0].get_average_processing_time())
        if self.sequencing_rule == "WSPT":
            return sorted(
                batch_job_list,
                key=lambda j: j.operation_list[0].get_average_processing_time() * getattr(j, "weight", 1.0),
            )
        if self.sequencing_rule == "JOHNSON":
            return johnson(batch_job_list)
        if self.sequencing_rule == "PALMER":
            return palmer(batch_job_list)
        if self.sequencing_rule == "RANDOM":
            shuffled = batch_job_list[:]
            rd.shuffle(shuffled)
            return shuffled
        return batch_job_list

    def sequence_jobs(self, batch_job_list: List[Job]):
        default_order = self._default_sequence(batch_job_list)

        if self.decision_manager is not None:
            state = {
                "source_id": self.id,
                "sequencing_rule": self.sequencing_rule,
                "candidate_jobs": [
                    {
                        "job_id": job.id,
                        "arrival_time": float(job.arrival_time),
                        "remaining_ops": len(job.operation_list) - job.step,
                        "remaining_work": self._total_avg_time(job),
                        "first_op_avg_time": float(job.operation_list[0].get_average_processing_time()),
                    }
                    for job in batch_job_list
                ],
            }
            selected_job_id = yield self.decision_manager.submit(
                decision_type="sequencing",
                local_state=state,
                candidates=[job.id for job in batch_job_list],
                default_action=default_order[0].id if default_order else None,
            )
            selected = next((job for job in default_order if job.id == selected_job_id), None)
            if selected is None:
                return default_order
            return [selected] + [job for job in default_order if job.id != selected.id]

        return self.sequencing(batch_job_list)

    def to_next_process(self, job: Job):
        next_operation = job.current_operation
        next_process_id = next_operation.process_list[0]
        job.last_operation_ready_time = float(self.env.now)

        print(f"{self.env.now:.2f}: Job {job.id} (Op: {next_operation.id}) first -> Process {next_process_id}")
        yield self.model[next_process_id].job_queue.put(job)
        self.monitor.record(
            time=self.env.now,
            part_id=job.id,
            operation=next_operation.id,
            process=next_process_id,
            machine=None,
            event="Job Transferred",
        )

    def sequencing(self, batch_job_list: List[Job]) -> List[Job]:
        default_order = self._default_sequence(batch_job_list)
        if self.decision_policy is None:
            return default_order

        state = {
            "decision_type": "sequencing",
            "time": float(self.env.now),
            "source_id": self.id,
            "sequencing_rule": self.sequencing_rule,
            "candidate_jobs": [
                {
                    "job_id": j.id,
                    "remaining_ops": len(j.operation_list) - j.step,
                    "first_op_avg_time": j.operation_list[0].get_average_processing_time(),
                }
                for j in batch_job_list
            ],
        }
        chosen_order_ids = self.decision_policy.choose_sequence(
            state=state,
            candidate_job_ids=[j.id for j in batch_job_list],
            default_order=[j.id for j in default_order],
        )
        if not chosen_order_ids:
            return default_order

        order_rank = {job_id: idx for idx, job_id in enumerate(chosen_order_ids)}
        return sorted(default_order, key=lambda j: order_rank.get(j.id, len(order_rank)))
