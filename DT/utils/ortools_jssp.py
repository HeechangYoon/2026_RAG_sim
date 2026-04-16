from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class OperationSchedule:
    job_id: str
    operation_id: str
    machine_type: str
    start: int
    end: int
    duration: int
    op_index: int


def solve_jssp_schedule(
    problem_data: dict[str, Any],
    time_limit_sec: float | None = None,
    require_optimal: bool = False,
) -> dict[str, Any]:
    """
    Solve a standard JSSP instance with OR-Tools CP-SAT and return a replayable schedule.

    This implementation assumes each operation has exactly one machine option, which is
    the format used by the benchmark JSSP instances in this project.
    """

    job_info = problem_data.get("job_info", {})
    operation_info = problem_data.get("operation_info", {})
    if not job_info or not operation_info:
        raise ValueError("problem_data is missing job_info or operation_info")

    jobs: list[tuple[str, float, list[tuple[str, str, int]]]] = []
    horizon = 0
    for job_id, info in job_info.items():
        arrival_time = float(info.get("arrival_time", 0.0) or 0.0)
        operations: list[tuple[str, str, int]] = []
        for op_id in info.get("operations", []):
            op = operation_info[op_id]
            machine_list = list(op.get("machine", []))
            proc_list = list(op.get("processing_time", []))
            if len(machine_list) != 1 or len(proc_list) != 1:
                raise NotImplementedError(
                    "OR-Tools replay teacher currently supports standard JSSP operations "
                    "with exactly one machine option per operation."
                )
            machine_type = str(machine_list[0])
            duration = int(round(float(proc_list[0])))
            operations.append((str(op_id), machine_type, duration))
            horizon += duration
        jobs.append((str(job_id), arrival_time, operations))

    model = cp_model.CpModel()
    task_vars: dict[tuple[str, int], tuple[cp_model.IntVar, cp_model.IntVar, cp_model.IntervalVar]] = {}
    machine_to_intervals: dict[str, list[cp_model.IntervalVar]] = {}

    for job_id, arrival_time, operations in jobs:
        previous_end = None
        release_lb = int(round(arrival_time))
        for op_index, (_, machine_type, duration) in enumerate(operations):
            suffix = f"{job_id}_{op_index}"
            start = model.NewIntVar(release_lb, horizon, f"start_{suffix}")
            end = model.NewIntVar(release_lb, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, duration, end, f"interval_{suffix}")
            task_vars[(job_id, op_index)] = (start, end, interval)
            machine_to_intervals.setdefault(machine_type, []).append(interval)
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

    for intervals in machine_to_intervals.values():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    last_ends = []
    for job_id, _, operations in jobs:
        _, end, _ = task_vars[(job_id, len(operations) - 1)]
        last_ends.append(end)
    model.AddMaxEquality(makespan, last_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    if time_limit_sec is not None and time_limit_sec > 0:
        solver.parameters.max_time_in_seconds = float(time_limit_sec)

    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    feasible_statuses = {cp_model.OPTIMAL, cp_model.FEASIBLE}
    if status not in feasible_statuses:
        raise RuntimeError(f"OR-Tools could not find a feasible solution. status={status_name}")
    if require_optimal and status != cp_model.OPTIMAL:
        raise RuntimeError(
            f"OR-Tools found only a non-optimal solution. status={status_name}"
        )

    instance_name = str(problem_data.get("instance_name") or problem_data.get("name") or "unknown")
    objective_value = float(solver.ObjectiveValue())
    wall_time = float(solver.WallTime())
    print(
        "[OR_TOOLS]"
        f" instance={instance_name}"
        f" status={status_name}"
        f" objective={objective_value:.3f}"
        f" optimal={status == cp_model.OPTIMAL}"
        f" wall_time_sec={wall_time:.3f}"
    )

    operations_by_id: dict[str, OperationSchedule] = {}
    job_to_operations: dict[str, list[OperationSchedule]] = {}
    for job_id, _, operations in jobs:
        scheduled_ops: list[OperationSchedule] = []
        for op_index, (op_id, machine_type, duration) in enumerate(operations):
            start, end, _ = task_vars[(job_id, op_index)]
            row = OperationSchedule(
                job_id=job_id,
                operation_id=op_id,
                machine_type=machine_type,
                start=int(solver.Value(start)),
                end=int(solver.Value(end)),
                duration=int(duration),
                op_index=op_index,
            )
            operations_by_id[op_id] = row
            scheduled_ops.append(row)
        job_to_operations[job_id] = scheduled_ops

    return {
        "status": status_name,
        "is_optimal": status == cp_model.OPTIMAL,
        "objective_value": objective_value,
        "wall_time_sec": wall_time,
        "operations_by_id": operations_by_id,
        "job_to_operations": job_to_operations,
    }
