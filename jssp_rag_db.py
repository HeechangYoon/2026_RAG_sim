import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from DT.agents import HeuristicSchedulingAgent, OptimalReplaySchedulingAgent
from DT.modeling import load_data_from_unified_csv, run_simulation_stepwise
from DT.utils.benchmarking_converter import convert_benchmarking_data


# Default heuristic grids used when user does not pass CLI overrides.
DEFAULT_SEQUENCING_RULES = ("FIFO", "SPT", "LPT", "FSPT", "RANDOM")
DEFAULT_DISPATCHING_RULES = ("FIFO", "SPT", "LPT", "MWKR", "LWKR", "RANDOM")
DEFAULT_ROUTING_RULE = "FIFO"
DEFAULT_LOG_CHUNK_SIZE = 120
DEFAULT_DECISION_MEMORY_TOP = 3
DEFAULT_DECISION_CASE_TOP = 5
DEFAULT_TEACHER_DISPATCHING_RULES = ("SPT",)
DEFAULT_TEACHER_SOURCE = "heuristic"
ORTOOLS_POLICY = None


@dataclass(frozen=True)
class HeuristicPolicy:
    sequencing_rule: str
    routing_rule: str
    dispatching_rule: str


def _teacher_tier_for_policy(
    policy: HeuristicPolicy,
    teacher_dispatching_rules: set[str],
    teacher_source: str = DEFAULT_TEACHER_SOURCE,
) -> str:
    if teacher_source == "ortools":
        return "reference"
    return "teacher" if policy.dispatching_rule in teacher_dispatching_rules else "reference"


def _refresh_teacher_tiers_for_instance(
    conn: sqlite3.Connection,
    instance_id: int,
    mode: str,
    teacher_dispatching_rules: set[str],
    teacher_top_k: int,
    teacher_source: str = DEFAULT_TEACHER_SOURCE,
) -> None:
    conn.execute("UPDATE runs SET teacher_tier = 'reference' WHERE instance_id = ?", (instance_id,))
    conn.execute(
        """
        UPDATE decision_cases
        SET teacher_tier = 'reference'
        WHERE run_id IN (SELECT run_id FROM runs WHERE instance_id = ?)
        """,
        (instance_id,),
    )

    if mode == "dispatch_rule":
        if teacher_source == "ortools":
            conn.execute(
                """
                UPDATE runs
                SET teacher_tier = 'teacher'
                WHERE instance_id = ?
                  AND sequencing_rule = 'OR_TOOLS'
                  AND routing_rule = 'OR_TOOLS'
                  AND dispatching_rule = 'OR_TOOLS'
                """,
                (instance_id,),
            )
        else:
            conn.execute(
                f"""
                UPDATE runs
                SET teacher_tier = 'teacher'
                WHERE instance_id = ?
                  AND dispatching_rule IN ({",".join("?" for _ in teacher_dispatching_rules)})
                """,
                (instance_id, *sorted(teacher_dispatching_rules)),
            )
    elif mode == "best_run":
        run_rows = conn.execute(
            """
            SELECT run_id
            FROM runs
            WHERE instance_id = ?
              AND status = 'success'
              AND makespan IS NOT NULL
            ORDER BY makespan ASC, run_id ASC
            LIMIT ?
            """,
            (instance_id, max(teacher_top_k, 1)),
        ).fetchall()
        for row in run_rows:
            conn.execute("UPDATE runs SET teacher_tier = 'teacher' WHERE run_id = ?", (int(row[0]),))
    else:
        raise ValueError(f"Unsupported teacher mode: {mode}")

    conn.execute(
        """
        UPDATE decision_cases
        SET teacher_tier = (
            SELECT runs.teacher_tier
            FROM runs
            WHERE runs.run_id = decision_cases.run_id
        )
        WHERE run_id IN (SELECT run_id FROM runs WHERE instance_id = ?)
        """,
        (instance_id,),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _split_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _list_jssp_instances(raw_jssp_dir: Path) -> list[str]:
    if not raw_jssp_dir.exists():
        return []
    names = set()
    for path in raw_jssp_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix == ".txt":
            names.add(path.stem)
        elif path.suffix == "":
            names.add(path.name)
    return sorted(names)


def _resolve_raw_jssp_source(raw_jssp_dir: Path, instance_name: str) -> Path:
    candidates = [raw_jssp_dir / instance_name, raw_jssp_dir / f"{instance_name}.txt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_problem_with_optional_convert(
    instance_name: str,
    dt_root: Path,
    preprocessed_dir: Path,
    raw_jssp_dir: Path,
    force_convert: bool,
) -> tuple[dict, Path]:
    # The simulator consumes unified CSV. If it does not exist (or force flag is set),
    # convert from the original benchmark .txt first.
    preprocessed_csv = preprocessed_dir / f"problem_{instance_name}.csv"
    raw_txt = _resolve_raw_jssp_source(raw_jssp_dir, instance_name)

    if force_convert or not preprocessed_csv.exists():
        if not raw_txt.exists():
            raise FileNotFoundError(
                f"Cannot find input files for {instance_name}. "
                f"Expected either {preprocessed_csv} or {raw_txt}"
            )
        convert_benchmarking_data(str(raw_txt), str(preprocessed_dir), "JSSP")

    data = load_data_from_unified_csv(str(preprocessed_dir), instance_name)
    if not data:
        raise RuntimeError(f"Failed to load problem data for {instance_name}")
    return data, preprocessed_csv


def _extract_operation_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(
            columns=[
                "job_id",
                "operation_id",
                "machine_instance",
                "machine_type",
                "start_time",
                "finish_time",
                "duration",
            ]
        )

    df = event_df.copy()
    df["event"] = df["event"].astype(str).str.lower()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    # Operation timing is derived by pairing:
    # "job assigned" -> start, "operation complete" -> finish.
    assigned = df[df["event"] == "job assigned"].copy()
    complete = df[df["event"] == "operation complete"].copy()
    if assigned.empty or complete.empty:
        return pd.DataFrame(
            columns=[
                "job_id",
                "operation_id",
                "machine_instance",
                "machine_type",
                "start_time",
                "finish_time",
                "duration",
            ]
        )

    assigned = assigned.rename(
        columns={
            "part_id": "job_id",
            "operation": "operation_id",
            "process": "machine_instance",
            "machine": "machine_type",
            "time": "start_time",
        }
    )
    complete = complete.rename(
        columns={
            "part_id": "job_id",
            "operation": "operation_id",
            "process": "machine_instance",
            "machine": "machine_type",
            "time": "finish_time",
        }
    )

    key_cols = ["job_id", "operation_id", "machine_instance"]
    assigned = assigned.sort_values("start_time")
    complete = complete.sort_values("finish_time")
    # Same key may appear multiple times in noisy logs. seq index makes pairing stable.
    assigned["seq"] = assigned.groupby(key_cols).cumcount()
    complete["seq"] = complete.groupby(key_cols).cumcount()

    merged = assigned.merge(
        complete[key_cols + ["seq", "finish_time"]],
        on=key_cols + ["seq"],
        how="left",
    )
    merged["duration"] = merged["finish_time"] - merged["start_time"]
    merged = merged.drop(columns=["seq"])
    return merged[
        [
            "job_id",
            "operation_id",
            "machine_instance",
            "machine_type",
            "start_time",
            "finish_time",
            "duration",
        ]
    ]


def _extract_job_completion(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(columns=["job_id", "completion_time"])

    df = event_df.copy()
    df["event"] = df["event"].astype(str).str.lower()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    completed = df[df["event"] == "job completed"]
    if completed.empty:
        return pd.DataFrame(columns=["job_id", "completion_time"])

    result = (
        completed.groupby("part_id", as_index=False)["time"]
        .max()
        .rename(columns={"part_id": "job_id", "time": "completion_time"})
    )
    return result


def _extract_machine_metrics(operation_df: pd.DataFrame, makespan: float) -> pd.DataFrame:
    if operation_df.empty:
        return pd.DataFrame(
            columns=["machine_instance", "machine_type", "busy_time", "utilization", "operations_count"]
        )
    grouped = (
        operation_df.groupby(["machine_instance", "machine_type"], as_index=False)
        .agg(
            busy_time=("duration", "sum"),
            operations_count=("operation_id", "count"),
        )
        .fillna(0.0)
    )
    # Utilization is normalized by run makespan to make runs comparable.
    if makespan > 0:
        grouped["utilization"] = grouped["busy_time"] / makespan
    else:
        grouped["utilization"] = 0.0
    return grouped


def _compute_makespan(event_df: pd.DataFrame, job_completion_df: pd.DataFrame) -> float:
    # Prefer completion-based makespan for correctness.
    # If completion events are missing, fallback to max event timestamp.
    if not job_completion_df.empty:
        return float(job_completion_df["completion_time"].max())
    if event_df.empty:
        return 0.0
    times = pd.to_numeric(event_df["time"], errors="coerce").dropna()
    return float(times.max()) if not times.empty else 0.0


def _build_rag_document(
    instance_name: str,
    policy: HeuristicPolicy,
    makespan: float,
    operation_df: pd.DataFrame,
    machine_df: pd.DataFrame,
    job_completion_df: pd.DataFrame,
) -> tuple[str, str, str]:
    total_jobs = len(job_completion_df)
    avg_completion = float(job_completion_df["completion_time"].mean()) if total_jobs else 0.0
    std_completion = float(job_completion_df["completion_time"].std(ddof=0)) if total_jobs else 0.0

    # Keep document compact and retrieval-friendly: summary + top bottleneck machines.
    bottleneck = machine_df.sort_values("utilization", ascending=False).head(3)
    bottleneck_text = "; ".join(
        f"{r.machine_instance} util={r.utilization:.3f}, busy={r.busy_time:.1f}"
        for r in bottleneck.itertuples(index=False)
    )
    if not bottleneck_text:
        bottleneck_text = "N/A"

    title = (
        f"{instance_name} | seq={policy.sequencing_rule}, "
        f"route={policy.routing_rule}, dispatch={policy.dispatching_rule}"
    )
    content = (
        f"Instance {instance_name} run summary. "
        f"Sequencing={policy.sequencing_rule}, Routing={policy.routing_rule}, "
        f"Dispatching={policy.dispatching_rule}. "
        f"Makespan={makespan:.3f}. Completed jobs={total_jobs}. "
        f"Average completion time={avg_completion:.3f}. Completion std={std_completion:.3f}. "
        f"Recorded operations={len(operation_df)}. "
        f"Bottleneck machines: {bottleneck_text}."
    )
    metadata = {
        "instance_name": instance_name,
        "sequencing_rule": policy.sequencing_rule,
        "routing_rule": policy.routing_rule,
        "dispatching_rule": policy.dispatching_rule,
        "makespan": makespan,
        "completed_jobs": total_jobs,
    }
    return title, content, json.dumps(metadata, ensure_ascii=True)


def _safe_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _build_decision_memories(
    instance_name: str,
    policy: HeuristicPolicy,
    makespan: float,
    problem_data: dict,
    operation_df: pd.DataFrame,
    machine_df: pd.DataFrame,
    job_completion_df: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    job_info = problem_data.get("job_info", {})
    operation_info = problem_data.get("operation_info", {})

    top_machines = machine_df.sort_values("utilization", ascending=False).head(DEFAULT_DECISION_MEMORY_TOP)
    bottleneck_text = "; ".join(
        f"{row.machine_type}:{row.machine_instance}:util={_safe_float(row.utilization):.3f}:busy={_safe_float(row.busy_time):.1f}"
        for row in top_machines.itertuples(index=False)
    ) or "none"

    job_totals = []
    for job_id, info in job_info.items():
        total_proc = 0.0
        for op_id in info.get("operations", []):
            op = operation_info.get(op_id, {})
            proc_times = op.get("processing_time", [])
            if proc_times:
                total_proc += sum(proc_times) / len(proc_times)
        job_totals.append((job_id, total_proc, len(info.get("operations", []))))
    shortest_jobs = sorted(job_totals, key=lambda item: item[1])[:DEFAULT_DECISION_MEMORY_TOP]
    longest_jobs = sorted(job_totals, key=lambda item: item[1], reverse=True)[:DEFAULT_DECISION_MEMORY_TOP]
    completion_rank = []
    if not job_completion_df.empty:
        completion_rank = sorted(
            (
                (str(row.job_id), _safe_float(row.completion_time))
                for row in job_completion_df.itertuples(index=False)
            ),
            key=lambda item: item[1],
        )[:DEFAULT_DECISION_MEMORY_TOP]

    machine_operation_stats = []
    if not operation_df.empty:
        op_stats = (
            operation_df.groupby("machine_type", as_index=False)
            .agg(
                avg_duration=("duration", "mean"),
                min_duration=("duration", "min"),
                max_duration=("duration", "max"),
                operations=("operation_id", "count"),
            )
            .sort_values("avg_duration")
            .head(DEFAULT_DECISION_MEMORY_TOP)
        )
        machine_operation_stats = [
            (
                str(row.machine_type),
                _safe_float(row.avg_duration),
                _safe_float(row.min_duration),
                _safe_float(row.max_duration),
                int(row.operations),
            )
            for row in op_stats.itertuples(index=False)
        ]

    sequencing_content = (
        f"decision_type=sequencing; instance={instance_name}; seq={policy.sequencing_rule}; "
        f"dispatch={policy.dispatching_rule}; route={policy.routing_rule}; makespan={makespan:.3f}; "
        f"jobs={len(job_info)}; bottlenecks={bottleneck_text}; "
        f"short_jobs={shortest_jobs}; long_jobs={longest_jobs}; "
        f"fast_finish={completion_rank}"
    )
    dispatch_content = (
        f"decision_type=dispatch; instance={instance_name}; seq={policy.sequencing_rule}; "
        f"dispatch={policy.dispatching_rule}; route={policy.routing_rule}; makespan={makespan:.3f}; "
        f"bottlenecks={bottleneck_text}; machine_op_stats={machine_operation_stats}; "
        f"fast_finish={completion_rank}"
    )
    routing_content = (
        f"decision_type=routing; instance={instance_name}; seq={policy.sequencing_rule}; "
        f"dispatch={policy.dispatching_rule}; route={policy.routing_rule}; makespan={makespan:.3f}; "
        f"bottlenecks={bottleneck_text}; machine_op_stats={machine_operation_stats}"
    )

    def metadata_json(decision_type: str) -> str:
        return json.dumps(
            {
                "instance_name": instance_name,
                "decision_type": decision_type,
                "sequencing_rule": policy.sequencing_rule,
                "routing_rule": policy.routing_rule,
                "dispatching_rule": policy.dispatching_rule,
                "makespan": makespan,
            },
            ensure_ascii=True,
        )

    return [
        ("sequencing", sequencing_content, metadata_json("sequencing")),
        ("dispatch", dispatch_content, metadata_json("dispatch")),
        ("routing", routing_content, metadata_json("routing")),
    ]


def _serialize_event_log(event_df: pd.DataFrame) -> str:
    if event_df is None or event_df.empty:
        return ""
    return event_df.to_csv(index=False)


def _build_log_chunks(event_df: pd.DataFrame, chunk_size: int) -> list[tuple[int, str]]:
    if event_df is None or event_df.empty:
        return []

    # Keep per-row event detail and split by fixed-size windows for retrieval.
    # Each line is compact but still human/LLM readable.
    lines: list[str] = []
    for row in event_df.itertuples(index=False):
        lines.append(
            f"time={row.time}, part_id={row.part_id}, operation={row.operation}, "
            f"process={row.process}, machine={row.machine}, event={row.event}"
        )

    chunks: list[tuple[int, str]] = []
    for i in range(0, len(lines), chunk_size):
        chunk_index = i // chunk_size
        chunk_text = "\n".join(lines[i : i + chunk_size])
        chunks.append((chunk_index, chunk_text))
    return chunks


def _build_dispatch_decision_cases(
    instance_name: str,
    policy: HeuristicPolicy,
    makespan: float,
    event_df: pd.DataFrame,
    operation_df: pd.DataFrame,
    machine_df: pd.DataFrame,
) -> list[tuple]:
    if event_df is None or event_df.empty:
        return []

    df = event_df.copy()
    df["event"] = df["event"].astype(str).str.lower()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    assigned = df[df["event"].eq("job assigned")].copy()
    if assigned.empty:
        return []

    op_lookup = {}
    if operation_df is not None and not operation_df.empty:
        for row in operation_df.itertuples(index=False):
            op_lookup[(str(row.job_id), str(row.operation_id))] = {
                "duration": _safe_float(row.duration),
                "finish_time": _safe_float(row.finish_time),
                "machine_type": str(row.machine_type) if pd.notna(row.machine_type) else None,
            }

    top_machines = machine_df.sort_values("utilization", ascending=False).head(DEFAULT_DECISION_CASE_TOP)
    bottleneck_summary = [
        {
            "machine_type": str(row.machine_type),
            "machine_instance": str(row.machine_instance),
            "utilization": _safe_float(row.utilization),
            "busy_time": _safe_float(row.busy_time),
        }
        for row in top_machines.itertuples(index=False)
    ]

    cases = []
    for row in assigned.itertuples(index=False):
        if pd.isna(row.machine) or pd.isna(row.part_id):
            continue
        case_time = _safe_float(row.time)
        machine_type = str(row.machine)
        chosen_job_id = str(row.part_id)
        operation_id = str(row.operation) if pd.notna(row.operation) else ""

        same_machine = assigned[
            (assigned["machine"] == machine_type)
            & (assigned["time"] == case_time)
        ].copy()
        same_machine = same_machine.sort_values("part_id")
        candidate_jobs = []
        for candidate in same_machine.itertuples(index=False):
            candidate_job_id = str(candidate.part_id)
            candidate_op_id = str(candidate.operation) if pd.notna(candidate.operation) else ""
            lookup = op_lookup.get((candidate_job_id, candidate_op_id), {})
            candidate_jobs.append(
                {
                    "job_id": candidate_job_id,
                    "operation_id": candidate_op_id,
                    "proc_time_on_machine": lookup.get("duration"),
                }
            )

        state_summary = {
            "sim_time": case_time,
            "machine_type": machine_type,
            "chosen_job_id": chosen_job_id,
            "candidate_jobs": candidate_jobs,
            "policy_dispatching_rule": policy.dispatching_rule,
            "bottleneck_summary": bottleneck_summary,
        }
        local_features = {
            "candidate_count": len(candidate_jobs),
            "machine_type": machine_type,
            "chosen_job_id": chosen_job_id,
            "chosen_proc_time": op_lookup.get((chosen_job_id, operation_id), {}).get("duration"),
        }
        outcome = {
            "final_makespan": makespan,
            "chosen_finish_time": op_lookup.get((chosen_job_id, operation_id), {}).get("finish_time"),
        }
        case_text = (
            f"decision_type=dispatch; instance={instance_name}; time={case_time:.3f}; "
            f"machine_type={machine_type}; chosen_job={chosen_job_id}; "
            f"dispatch_rule={policy.dispatching_rule}; makespan={makespan:.3f}; "
            f"candidates={candidate_jobs}; bottlenecks={bottleneck_summary}"
        )
        cases.append(
            (
                instance_name,
                "dispatch",
                case_time,
                machine_type,
                chosen_job_id,
                policy.dispatching_rule,
                json.dumps(state_summary, ensure_ascii=True),
                json.dumps(local_features, ensure_ascii=True),
                json.dumps(outcome, ensure_ascii=True),
                case_text,
                json.dumps(
                    {
                        "instance_name": instance_name,
                        "decision_type": "dispatch",
                        "machine_type": machine_type,
                        "dispatching_rule": policy.dispatching_rule,
                        "makespan": makespan,
                    },
                    ensure_ascii=True,
                ),
            )
        )
    return cases


def _load_decision_trace(trace_path: Path) -> list[dict]:
    if not trace_path.exists():
        return []
    rows: list[dict] = []
    with trace_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _build_decision_cases_from_trace(
    instance_name: str,
    policy: HeuristicPolicy,
    makespan: float,
    decision_rows: list[dict],
    operation_df: pd.DataFrame,
    machine_df: pd.DataFrame,
) -> list[tuple]:
    if not decision_rows:
        return []

    op_lookup = {}
    if operation_df is not None and not operation_df.empty:
        for row in operation_df.itertuples(index=False):
            op_lookup[(str(row.job_id), str(row.operation_id))] = {
                "duration": _safe_float(row.duration),
                "finish_time": _safe_float(row.finish_time),
                "machine_type": str(row.machine_type) if pd.notna(row.machine_type) else None,
            }

    top_machines = machine_df.sort_values("utilization", ascending=False).head(DEFAULT_DECISION_CASE_TOP)
    bottleneck_summary = [
        {
            "machine_type": str(row.machine_type),
            "machine_instance": str(row.machine_instance),
            "utilization": _safe_float(row.utilization),
            "busy_time": _safe_float(row.busy_time),
        }
        for row in top_machines.itertuples(index=False)
    ]

    cases: list[tuple] = []
    for row in decision_rows:
        if str(row.get("decision_type", "")).strip().lower() != "dispatch":
            continue

        state = row.get("state", {})
        if not isinstance(state, dict):
            state = {}
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []

        sim_time = _safe_float(state.get("sim_time"))
        machine_type = str(state.get("machine_type", "") or "")
        chosen_action = row.get("parsed_choice")
        default_action = row.get("default_action")
        candidate_jobs = state.get("candidate_jobs", [])
        if not isinstance(candidate_jobs, list):
            candidate_jobs = []

        candidate_count = len(candidates)
        proc_times = []
        remaining_work_values = []
        chosen_rank_by_proc = None
        chosen_rank_by_remaining_work = None
        global_job_summary = state.get("global_job_summary", [])
        if not isinstance(global_job_summary, list):
            global_job_summary = []
        global_remaining_work_values = [
            _safe_float(item.get("remaining_work"))
            for item in global_job_summary
            if isinstance(item, dict)
        ]
        for candidate in candidate_jobs:
            if not isinstance(candidate, dict):
                continue
            proc_times.append(_safe_float(candidate.get("proc_time_on_machine")))
            remaining_work_values.append(_safe_float(candidate.get("remaining_work")))

        sortable_proc = []
        sortable_remaining = []
        for candidate in candidate_jobs:
            if not isinstance(candidate, dict):
                continue
            sortable_proc.append(
                (
                    str(candidate.get("job_id")),
                    _safe_float(candidate.get("proc_time_on_machine")),
                )
            )
            sortable_remaining.append(
                (
                    str(candidate.get("job_id")),
                    _safe_float(candidate.get("remaining_work")),
                )
            )
        sortable_proc.sort(key=lambda item: (item[1], item[0]))
        sortable_remaining.sort(key=lambda item: (item[1], item[0]))
        chosen_action_str = str(chosen_action)
        for idx, (job_id, _) in enumerate(sortable_proc):
            if job_id == chosen_action_str:
                chosen_rank_by_proc = idx
                break
        for idx, (job_id, _) in enumerate(sortable_remaining):
            if job_id == chosen_action_str:
                chosen_rank_by_remaining_work = idx
                break
        candidate_profile_vector = sorted(
            [
                (
                    _safe_float(candidate.get("proc_time_on_machine")),
                    _safe_float(candidate.get("remaining_work")),
                )
                for candidate in candidate_jobs
                if isinstance(candidate, dict)
            ],
            key=lambda item: (item[0], item[1]),
        )

        chosen_finish_time = None
        for candidate in candidate_jobs:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("job_id")) != str(chosen_action):
                continue
            op_key = (str(candidate.get("job_id")), str(candidate.get("operation_id", "")))
            chosen_finish_time = op_lookup.get(op_key, {}).get("finish_time")
            break

        state_summary = {
            "sim_time": sim_time,
            "machine_type": machine_type,
            "chosen_action": chosen_action,
            "default_action": default_action,
            "candidates": candidates,
            "candidate_jobs": candidate_jobs,
            "problem_summary": state.get("problem_summary", {}),
            "global_job_summary": state.get("global_job_summary", []),
            "bottleneck_summary": state.get("bottleneck_machines", bottleneck_summary),
        }
        local_features = {
            "candidate_count": candidate_count,
            "avg_proc_time": (sum(proc_times) / len(proc_times)) if proc_times else 0.0,
            "avg_remaining_work": (sum(remaining_work_values) / len(remaining_work_values))
            if remaining_work_values
            else 0.0,
            "min_proc_time": min(proc_times) if proc_times else 0.0,
            "max_proc_time": max(proc_times) if proc_times else 0.0,
            "min_remaining_work": min(remaining_work_values) if remaining_work_values else 0.0,
            "max_remaining_work": max(remaining_work_values) if remaining_work_values else 0.0,
            "candidate_profile_vector": candidate_profile_vector,
            "global_remaining_work_vector": sorted(global_remaining_work_values, reverse=True)[:10],
            "global_remaining_jobs": len(global_remaining_work_values),
            "global_avg_remaining_work": (
                sum(global_remaining_work_values) / len(global_remaining_work_values)
                if global_remaining_work_values
                else 0.0
            ),
            "global_max_remaining_work": max(global_remaining_work_values) if global_remaining_work_values else 0.0,
            "bottleneck_queue": max(
                [
                    _safe_float(item.get("queue_size"))
                    for item in state_summary["bottleneck_summary"]
                    if isinstance(item, dict)
                ]
                or [0.0]
            ),
            "chosen_rank_by_proc": chosen_rank_by_proc,
            "chosen_rank_by_remaining_work": chosen_rank_by_remaining_work,
        }
        # Retrieval cases should depend on structured state/action features that
        # the system can derive deterministically from the simulator state.
        # Natural-language reasons are optional trace artifacts, not core case
        # features, so keep them out of the case outcome payload.
        outcome = {
            "final_makespan": makespan,
            "chosen_finish_time": chosen_finish_time,
            "error": row.get("error"),
        }
        content = (
            f"decision_type=dispatch; instance={instance_name}; time={sim_time:.3f}; "
            f"machine_type={machine_type}; chosen_job={chosen_action}; default_action={default_action}; "
            f"dispatch_rule={policy.dispatching_rule}; makespan={makespan:.3f}; "
            f"candidates={candidate_jobs}; bottlenecks={state_summary['bottleneck_summary']}"
        )
        metadata = {
            "instance_name": instance_name,
            "decision_type": "dispatch",
            "machine_type": machine_type,
            "dispatching_rule": policy.dispatching_rule,
            "chosen_action": chosen_action,
            "default_action": default_action,
            "makespan": makespan,
        }
        cases.append(
            (
                instance_name,
                "dispatch",
                sim_time,
                machine_type,
                chosen_action,
                default_action,
                json.dumps(state_summary, ensure_ascii=True),
                json.dumps(local_features, ensure_ascii=True),
                json.dumps(outcome, ensure_ascii=True),
                content,
                json.dumps(metadata, ensure_ascii=True),
            )
        )
    return cases


def _build_instance_features(problem_data: dict, operation_df: pd.DataFrame, machine_df: pd.DataFrame) -> dict:
    job_info = problem_data.get("job_info", {})
    machine_info = problem_data.get("machine_info", {})
    operation_info = problem_data.get("operation_info", {})

    proc_values = []
    job_total_work = []
    for job_id, info in job_info.items():
        total_work = 0.0
        for op_id in info.get("operations", []):
            op = operation_info.get(op_id, {})
            proc_times = [float(v) for v in op.get("processing_time", [])]
            proc_values.extend(proc_times)
            if proc_times:
                total_work += sum(proc_times) / len(proc_times)
        job_total_work.append(total_work)

    bottleneck_util = 0.0
    util_std = 0.0
    if machine_df is not None and not machine_df.empty:
        util_series = machine_df["utilization"].astype(float)
        bottleneck_util = float(util_series.max())
        util_std = float(util_series.std(ddof=0))

    avg_proc = float(sum(proc_values) / len(proc_values)) if proc_values else 0.0
    proc_var = float(pd.Series(proc_values).std(ddof=0)) if proc_values else 0.0
    avg_job_work = float(sum(job_total_work) / len(job_total_work)) if job_total_work else 0.0
    job_work_std = float(pd.Series(job_total_work).std(ddof=0)) if job_total_work else 0.0

    return {
        "n_jobs": len(job_info),
        "n_machines": len(machine_info),
        "n_operations": len(operation_info),
        "avg_proc_time": avg_proc,
        "proc_time_std": proc_var,
        "avg_job_work": avg_job_work,
        "job_work_std": job_work_std,
        "bottleneck_utilization": bottleneck_util,
        "machine_util_std": util_std,
    }


def _init_db(conn: sqlite3.Connection) -> None:
    # Base relational schema: simulation facts + retrieval documents.
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS instances (
            instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            source_txt_path TEXT,
            preprocessed_csv_path TEXT,
            n_jobs INTEGER NOT NULL,
            n_machines INTEGER NOT NULL,
            n_operations INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id INTEGER NOT NULL,
            sequencing_rule TEXT NOT NULL,
            routing_rule TEXT NOT NULL,
            dispatching_rule TEXT NOT NULL,
            teacher_tier TEXT NOT NULL DEFAULT 'reference',
            random_seed INTEGER NOT NULL,
            status TEXT NOT NULL,
            makespan REAL,
            total_jobs INTEGER,
            completed_jobs INTEGER,
            event_log_path TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(instance_id, sequencing_rule, routing_rule, dispatching_rule, random_seed),
            FOREIGN KEY(instance_id) REFERENCES instances(instance_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS job_metrics (
            run_id INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            completion_time REAL,
            PRIMARY KEY (run_id, job_id),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS operation_metrics (
            run_id INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            machine_instance TEXT,
            machine_type TEXT,
            start_time REAL,
            finish_time REAL,
            duration REAL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS machine_metrics (
            run_id INTEGER NOT NULL,
            machine_instance TEXT NOT NULL,
            machine_type TEXT,
            busy_time REAL,
            utilization REAL,
            operations_count INTEGER,
            PRIMARY KEY (run_id, machine_instance),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS rag_documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            instance_name TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS run_logs (
            run_id INTEGER PRIMARY KEY,
            full_log_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS rag_log_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            instance_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS rag_decision_memories (
            memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            instance_name TEXT NOT NULL,
            decision_type TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS decision_cases (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            instance_name TEXT NOT NULL,
            decision_type TEXT NOT NULL,
            teacher_tier TEXT NOT NULL DEFAULT 'reference',
            sim_time REAL,
            machine_type TEXT,
            chosen_action TEXT,
            default_action TEXT,
            state_summary_json TEXT,
            local_features_json TEXT,
            outcome_json TEXT,
            content TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS decision_traces (
            trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            decision_index INTEGER NOT NULL,
            decision_type TEXT NOT NULL,
            sim_time REAL,
            state_json TEXT,
            candidates_json TEXT,
            default_action TEXT,
            chosen_action TEXT,
            reason TEXT,
            error TEXT,
            query_error TEXT,
            usage_json TEXT,
            raw_response TEXT,
            response_body TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS instance_features (
            run_id INTEGER PRIMARY KEY,
            instance_name TEXT NOT NULL,
            n_jobs INTEGER NOT NULL,
            n_machines INTEGER NOT NULL,
            n_operations INTEGER NOT NULL,
            avg_proc_time REAL,
            proc_time_std REAL,
            avg_job_work REAL,
            job_work_std REAL,
            bottleneck_utilization REAL,
            machine_util_std REAL,
            feature_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_runs_instance ON runs(instance_id);
        CREATE INDEX IF NOT EXISTS idx_runs_rules ON runs(sequencing_rule, routing_rule, dispatching_rule);
        CREATE INDEX IF NOT EXISTS idx_job_metrics_completion ON job_metrics(completion_time);
        CREATE INDEX IF NOT EXISTS idx_machine_metrics_util ON machine_metrics(utilization);
        CREATE INDEX IF NOT EXISTS idx_rag_log_chunks_run ON rag_log_chunks(run_id, chunk_index);
        CREATE INDEX IF NOT EXISTS idx_rag_decision_memories_run ON rag_decision_memories(run_id, decision_type);
        CREATE INDEX IF NOT EXISTS idx_decision_cases_run ON decision_cases(run_id, decision_type);
        CREATE INDEX IF NOT EXISTS idx_decision_traces_run ON decision_traces(run_id, decision_type, decision_index);
        CREATE INDEX IF NOT EXISTS idx_instance_features_name ON instance_features(instance_name);
        """
    )

    try:
        conn.execute("ALTER TABLE runs ADD COLUMN teacher_tier TEXT NOT NULL DEFAULT 'reference'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE decision_cases ADD COLUMN teacher_tier TEXT NOT NULL DEFAULT 'reference'")
    except sqlite3.OperationalError:
        pass

    # FTS5 is optional on some SQLite builds. If unavailable, keep DB usable without it.
    try:
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS rag_documents_fts USING fts5(
                title,
                content,
                metadata_json,
                content='rag_documents',
                content_rowid='doc_id'
            );

            CREATE TRIGGER IF NOT EXISTS rag_documents_ai AFTER INSERT ON rag_documents BEGIN
                INSERT INTO rag_documents_fts(rowid, title, content, metadata_json)
                VALUES (new.doc_id, new.title, new.content, new.metadata_json);
            END;

            CREATE TRIGGER IF NOT EXISTS rag_documents_ad AFTER DELETE ON rag_documents BEGIN
                INSERT INTO rag_documents_fts(rag_documents_fts, rowid, title, content, metadata_json)
                VALUES('delete', old.doc_id, old.title, old.content, old.metadata_json);
            END;

            CREATE TRIGGER IF NOT EXISTS rag_documents_au AFTER UPDATE ON rag_documents BEGIN
                INSERT INTO rag_documents_fts(rag_documents_fts, rowid, title, content, metadata_json)
                VALUES('delete', old.doc_id, old.title, old.content, old.metadata_json);
                INSERT INTO rag_documents_fts(rowid, title, content, metadata_json)
                VALUES (new.doc_id, new.title, new.content, new.metadata_json);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS rag_log_chunks_fts USING fts5(
                content,
                metadata_json,
                content='rag_log_chunks',
                content_rowid='chunk_id'
            );

            CREATE TRIGGER IF NOT EXISTS rag_log_chunks_ai AFTER INSERT ON rag_log_chunks BEGIN
                INSERT INTO rag_log_chunks_fts(rowid, content, metadata_json)
                VALUES (new.chunk_id, new.content, new.metadata_json);
            END;

            CREATE TRIGGER IF NOT EXISTS rag_log_chunks_ad AFTER DELETE ON rag_log_chunks BEGIN
                INSERT INTO rag_log_chunks_fts(rag_log_chunks_fts, rowid, content, metadata_json)
                VALUES('delete', old.chunk_id, old.content, old.metadata_json);
            END;

            CREATE TRIGGER IF NOT EXISTS rag_log_chunks_au AFTER UPDATE ON rag_log_chunks BEGIN
                INSERT INTO rag_log_chunks_fts(rag_log_chunks_fts, rowid, content, metadata_json)
                VALUES('delete', old.chunk_id, old.content, old.metadata_json);
                INSERT INTO rag_log_chunks_fts(rowid, content, metadata_json)
                VALUES (new.chunk_id, new.content, new.metadata_json);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS rag_decision_memories_fts USING fts5(
                decision_type,
                content,
                metadata_json,
                content='rag_decision_memories',
                content_rowid='memory_id'
            );

            CREATE TRIGGER IF NOT EXISTS rag_decision_memories_ai AFTER INSERT ON rag_decision_memories BEGIN
                INSERT INTO rag_decision_memories_fts(rowid, decision_type, content, metadata_json)
                VALUES (new.memory_id, new.decision_type, new.content, new.metadata_json);
            END;

            CREATE TRIGGER IF NOT EXISTS rag_decision_memories_ad AFTER DELETE ON rag_decision_memories BEGIN
                INSERT INTO rag_decision_memories_fts(rag_decision_memories_fts, rowid, decision_type, content, metadata_json)
                VALUES('delete', old.memory_id, old.decision_type, old.content, old.metadata_json);
            END;

            CREATE TRIGGER IF NOT EXISTS rag_decision_memories_au AFTER UPDATE ON rag_decision_memories BEGIN
                INSERT INTO rag_decision_memories_fts(rag_decision_memories_fts, rowid, decision_type, content, metadata_json)
                VALUES('delete', old.memory_id, old.decision_type, old.content, old.metadata_json);
                INSERT INTO rag_decision_memories_fts(rowid, decision_type, content, metadata_json)
                VALUES (new.memory_id, new.decision_type, new.content, new.metadata_json);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS decision_cases_fts USING fts5(
                decision_type,
                machine_type,
                content,
                metadata_json,
                content='decision_cases',
                content_rowid='case_id'
            );

            CREATE TRIGGER IF NOT EXISTS decision_cases_ai AFTER INSERT ON decision_cases BEGIN
                INSERT INTO decision_cases_fts(rowid, decision_type, machine_type, content, metadata_json)
                VALUES (new.case_id, new.decision_type, new.machine_type, new.content, new.metadata_json);
            END;

            CREATE TRIGGER IF NOT EXISTS decision_cases_ad AFTER DELETE ON decision_cases BEGIN
                INSERT INTO decision_cases_fts(decision_cases_fts, rowid, decision_type, machine_type, content, metadata_json)
                VALUES('delete', old.case_id, old.decision_type, old.machine_type, old.content, old.metadata_json);
            END;

            CREATE TRIGGER IF NOT EXISTS decision_cases_au AFTER UPDATE ON decision_cases BEGIN
                INSERT INTO decision_cases_fts(decision_cases_fts, rowid, decision_type, machine_type, content, metadata_json)
                VALUES('delete', old.case_id, old.decision_type, old.machine_type, old.content, old.metadata_json);
                INSERT INTO decision_cases_fts(rowid, decision_type, machine_type, content, metadata_json)
                VALUES (new.case_id, new.decision_type, new.machine_type, new.content, new.metadata_json);
            END;
            """
        )
    except sqlite3.OperationalError:
        pass

    conn.commit()


def _upsert_instance(
    conn: sqlite3.Connection,
    instance_name: str,
    source_txt_path: Path,
    preprocessed_csv_path: Path,
    problem_data: dict,
) -> int:
    # Instance metadata can be refreshed when conversion input changes.
    # Name is treated as the natural key.
    n_jobs = len(problem_data.get("job_info", {}))
    n_machines = len(problem_data.get("machine_info", {}))
    n_operations = len(problem_data.get("operation_info", {}))
    now = utc_now_iso()

    conn.execute(
        """
        INSERT INTO instances(
            name, source_txt_path, preprocessed_csv_path, n_jobs, n_machines, n_operations, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            source_txt_path=excluded.source_txt_path,
            preprocessed_csv_path=excluded.preprocessed_csv_path,
            n_jobs=excluded.n_jobs,
            n_machines=excluded.n_machines,
            n_operations=excluded.n_operations
        """,
        (
            instance_name,
            str(source_txt_path),
            str(preprocessed_csv_path),
            n_jobs,
            n_machines,
            n_operations,
            now,
        ),
    )
    row = conn.execute("SELECT instance_id FROM instances WHERE name = ?", (instance_name,)).fetchone()
    assert row is not None
    return int(row[0])


def _existing_run_id(
    conn: sqlite3.Connection,
    instance_id: int,
    policy: HeuristicPolicy,
    seed: int,
) -> int | None:
    # A run is uniquely identified by (instance, policy, seed).
    row = conn.execute(
        """
        SELECT run_id FROM runs
        WHERE instance_id = ?
          AND sequencing_rule = ?
          AND routing_rule = ?
          AND dispatching_rule = ?
          AND random_seed = ?
        """,
        (instance_id, policy.sequencing_rule, policy.routing_rule, policy.dispatching_rule, seed),
    ).fetchone()
    return int(row[0]) if row else None


def _insert_run_row(
    conn: sqlite3.Connection,
    instance_id: int,
    policy: HeuristicPolicy,
    teacher_tier: str,
    seed: int,
    status: str,
    makespan: float | None,
    total_jobs: int | None,
    completed_jobs: int | None,
    event_log_path: Path | None,
    error_message: str | None,
) -> int:
    now = utc_now_iso()
    cur = conn.execute(
        """
        INSERT INTO runs(
            instance_id, sequencing_rule, routing_rule, dispatching_rule, teacher_tier, random_seed, status,
            makespan, total_jobs, completed_jobs, event_log_path, error_message, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            instance_id,
            policy.sequencing_rule,
            policy.routing_rule,
            policy.dispatching_rule,
            teacher_tier,
            seed,
            status,
            makespan,
            total_jobs,
            completed_jobs,
            str(event_log_path) if event_log_path else None,
            error_message,
            now,
        ),
    )
    return int(cur.lastrowid)


def _delete_run_and_children(conn: sqlite3.Connection, run_id: int) -> None:
    # Child rows are cleaned by FK ON DELETE CASCADE.
    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def _insert_metrics(
    conn: sqlite3.Connection,
    run_id: int,
    instance_name: str,
    policy: HeuristicPolicy,
    makespan: float,
    operation_df: pd.DataFrame,
    machine_df: pd.DataFrame,
    job_completion_df: pd.DataFrame,
) -> None:
    # Bulk insert metric tables for throughput.
    if not job_completion_df.empty:
        conn.executemany(
            """
            INSERT INTO job_metrics(run_id, job_id, completion_time)
            VALUES (?, ?, ?)
            """,
            [
                (run_id, str(r.job_id), float(r.completion_time))
                for r in job_completion_df.itertuples(index=False)
            ],
        )

    if not operation_df.empty:
        conn.executemany(
            """
            INSERT INTO operation_metrics(
                run_id, job_id, operation_id, machine_instance, machine_type, start_time, finish_time, duration
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    str(r.job_id),
                    str(r.operation_id),
                    str(r.machine_instance) if pd.notna(r.machine_instance) else None,
                    str(r.machine_type) if pd.notna(r.machine_type) else None,
                    float(r.start_time) if pd.notna(r.start_time) else None,
                    float(r.finish_time) if pd.notna(r.finish_time) else None,
                    float(r.duration) if pd.notna(r.duration) else None,
                )
                for r in operation_df.itertuples(index=False)
            ],
        )

    if not machine_df.empty:
        conn.executemany(
            """
            INSERT INTO machine_metrics(
                run_id, machine_instance, machine_type, busy_time, utilization, operations_count
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    str(r.machine_instance),
                    str(r.machine_type) if pd.notna(r.machine_type) else None,
                    float(r.busy_time) if pd.notna(r.busy_time) else 0.0,
                    float(r.utilization) if pd.notna(r.utilization) else 0.0,
                    int(r.operations_count) if pd.notna(r.operations_count) else 0,
                )
                for r in machine_df.itertuples(index=False)
            ],
        )

    # One retrieval document per run keeps ranking and citation simpler.
    title, content, metadata_json = _build_rag_document(
        instance_name,
        policy,
        makespan,
        operation_df,
        machine_df,
        job_completion_df,
    )
    conn.execute(
        """
        INSERT INTO rag_documents(run_id, instance_name, title, content, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            instance_name=excluded.instance_name,
            title=excluded.title,
            content=excluded.content,
            metadata_json=excluded.metadata_json,
            created_at=excluded.created_at
        """,
        (run_id, instance_name, title, content, metadata_json, utc_now_iso()),
    )


def _insert_log_documents(
    conn: sqlite3.Connection,
    run_id: int,
    instance_name: str,
    policy: HeuristicPolicy,
    makespan: float,
    event_df: pd.DataFrame,
    chunk_size: int,
) -> None:
    full_log = _serialize_event_log(event_df)
    conn.execute(
        """
        INSERT INTO run_logs(run_id, full_log_text, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            full_log_text=excluded.full_log_text,
            created_at=excluded.created_at
        """,
        (run_id, full_log, utc_now_iso()),
    )

    conn.execute("DELETE FROM rag_log_chunks WHERE run_id = ?", (run_id,))
    chunks = _build_log_chunks(event_df, chunk_size)
    if not chunks:
        return

    conn.executemany(
        """
        INSERT INTO rag_log_chunks(
            run_id, instance_name, chunk_index, content, metadata_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                instance_name,
                chunk_index,
                content,
                json.dumps(
                    {
                        "instance_name": instance_name,
                        "sequencing_rule": policy.sequencing_rule,
                        "routing_rule": policy.routing_rule,
                        "dispatching_rule": policy.dispatching_rule,
                        "makespan": makespan,
                        "chunk_index": chunk_index,
                    },
                    ensure_ascii=True,
                ),
                utc_now_iso(),
            )
            for chunk_index, content in chunks
        ],
    )


def _insert_decision_memories(
    conn: sqlite3.Connection,
    run_id: int,
    instance_name: str,
    policy: HeuristicPolicy,
    makespan: float,
    problem_data: dict,
    operation_df: pd.DataFrame,
    machine_df: pd.DataFrame,
    job_completion_df: pd.DataFrame,
) -> None:
    conn.execute("DELETE FROM rag_decision_memories WHERE run_id = ?", (run_id,))
    memories = _build_decision_memories(
        instance_name=instance_name,
        policy=policy,
        makespan=makespan,
        problem_data=problem_data,
        operation_df=operation_df,
        machine_df=machine_df,
        job_completion_df=job_completion_df,
    )
    conn.executemany(
        """
        INSERT INTO rag_decision_memories(
            run_id, instance_name, decision_type, content, metadata_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (run_id, instance_name, decision_type, content, metadata_json, utc_now_iso())
            for decision_type, content, metadata_json in memories
        ],
    )


def _insert_decision_cases(
    conn: sqlite3.Connection,
    run_id: int,
    instance_name: str,
    policy: HeuristicPolicy,
    teacher_tier: str,
    makespan: float,
    decision_rows: list[dict],
    operation_df: pd.DataFrame,
    machine_df: pd.DataFrame,
) -> None:
    conn.execute("DELETE FROM decision_cases WHERE run_id = ?", (run_id,))
    dispatch_cases = _build_decision_cases_from_trace(
        instance_name=instance_name,
        policy=policy,
        makespan=makespan,
        decision_rows=decision_rows,
        operation_df=operation_df,
        machine_df=machine_df,
    )
    if not dispatch_cases:
        return
    conn.executemany(
        """
        INSERT INTO decision_cases(
            run_id, instance_name, decision_type, teacher_tier, sim_time, machine_type, chosen_action, default_action,
            state_summary_json, local_features_json, outcome_json, content, metadata_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                case[0],
                case[1],
                teacher_tier,
                case[2],
                case[3],
                case[4],
                case[5],
                case[6],
                case[7],
                case[8],
                case[9],
                case[10],
                utc_now_iso(),
            )
            for case in dispatch_cases
        ],
    )


def _insert_decision_traces(
    conn: sqlite3.Connection,
    run_id: int,
    decision_rows: list[dict],
) -> None:
    conn.execute("DELETE FROM decision_traces WHERE run_id = ?", (run_id,))
    if not decision_rows:
        return

    payloads = []
    for index, row in enumerate(decision_rows):
        state = row.get("state", {})
        if not isinstance(state, dict):
            state = {}
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []

        raw_response = row.get("raw_response")
        if raw_response is not None and not isinstance(raw_response, str):
            raw_response = json.dumps(raw_response, ensure_ascii=True)

        response_body = row.get("response_body")
        if response_body is not None:
            response_body = json.dumps(response_body, ensure_ascii=True)

        payloads.append(
            (
                run_id,
                index,
                str(row.get("decision_type", "") or ""),
                _safe_float(state.get("sim_time")),
                json.dumps(state, ensure_ascii=True),
                json.dumps(candidates, ensure_ascii=True),
                row.get("default_action"),
                row.get("parsed_choice"),
                row.get("reason"),
                row.get("error"),
                row.get("query_error"),
                json.dumps(row.get("usage"), ensure_ascii=True) if row.get("usage") is not None else None,
                raw_response,
                response_body,
                json.dumps(
                    {
                        "logged_at": row.get("logged_at"),
                    },
                    ensure_ascii=True,
                ),
                utc_now_iso(),
            )
        )

    conn.executemany(
        """
        INSERT INTO decision_traces(
            run_id, decision_index, decision_type, sim_time, state_json, candidates_json,
            default_action, chosen_action, reason, error, query_error, usage_json,
            raw_response, response_body, metadata_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payloads,
    )


def _insert_instance_features(
    conn: sqlite3.Connection,
    run_id: int,
    instance_name: str,
    problem_data: dict,
    operation_df: pd.DataFrame,
    machine_df: pd.DataFrame,
) -> None:
    features = _build_instance_features(problem_data, operation_df, machine_df)
    conn.execute("DELETE FROM instance_features WHERE run_id = ?", (run_id,))
    conn.execute(
        """
        INSERT INTO instance_features(
            run_id, instance_name, n_jobs, n_machines, n_operations,
            avg_proc_time, proc_time_std, avg_job_work, job_work_std,
            bottleneck_utilization, machine_util_std, feature_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            instance_name,
            int(features["n_jobs"]),
            int(features["n_machines"]),
            int(features["n_operations"]),
            float(features["avg_proc_time"]),
            float(features["proc_time_std"]),
            float(features["avg_job_work"]),
            float(features["job_work_std"]),
            float(features["bottleneck_utilization"]),
            float(features["machine_util_std"]),
            json.dumps(features, ensure_ascii=True),
            utc_now_iso(),
        ),
    )


def run_jssp_batch_to_db(
    dt_root: Path,
    db_path: Path,
    results_dir: Path,
    instance_names: Iterable[str],
    policies: Iterable[HeuristicPolicy],
    teacher_dispatching_rules: set[str],
    teacher_mode: str,
    teacher_top_k: int,
    teacher_source: str,
    ortools_max_time_sec: float | None,
    ortools_require_optimal: bool,
    seed: int,
    significant_digits: int,
    force_convert: bool,
    overwrite: bool,
    log_chunk_size: int = DEFAULT_LOG_CHUNK_SIZE,
) -> dict:
    preprocessed_dir = dt_root / "data" / "preprocessed"
    raw_jssp_dir = dt_root / "data" / "raw" / "JSSP"
    results_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    _init_db(conn)

    summary = {"success": 0, "failed": 0, "skipped": 0}

    for instance_name in instance_names:
        source_txt_path = _resolve_raw_jssp_source(raw_jssp_dir, instance_name)
        try:
            problem_data, preprocessed_csv = _load_problem_with_optional_convert(
                instance_name=instance_name,
                dt_root=dt_root,
                preprocessed_dir=preprocessed_dir,
                raw_jssp_dir=raw_jssp_dir,
                force_convert=force_convert,
            )
            instance_id = _upsert_instance(
                conn=conn,
                instance_name=instance_name,
                source_txt_path=source_txt_path,
                preprocessed_csv_path=preprocessed_csv,
                problem_data=problem_data,
            )
            conn.commit()
        except Exception as exc:
            print(f"[FAIL] instance={instance_name} load error: {exc}")
            summary["failed"] += 1
            continue

        run_specs: list[tuple[HeuristicPolicy, str, str]] = [
            (
                policy,
                _teacher_tier_for_policy(policy, teacher_dispatching_rules, teacher_source=teacher_source),
                "heuristic",
            )
            for policy in policies
        ]
        if teacher_source == "ortools":
            run_specs.append(
                (
                    HeuristicPolicy("OR_TOOLS", "OR_TOOLS", "OR_TOOLS"),
                    "teacher",
                    "ortools",
                )
            )

        for policy, teacher_tier, run_source in run_specs:
            existing_id = _existing_run_id(conn, instance_id, policy, seed)
            if existing_id is not None and not overwrite:
                print(
                    f"[SKIP] {instance_name} | seq={policy.sequencing_rule}, "
                    f"route={policy.routing_rule}, dispatch={policy.dispatching_rule} already exists"
                )
                summary["skipped"] += 1
                continue
            if existing_id is not None and overwrite:
                # Replace existing run atomically for deterministic reruns.
                _delete_run_and_children(conn, existing_id)
                conn.commit()

            # RANDOM-based heuristic paths become reproducible per run.
            random.seed(seed)
            log_path = (
                results_dir
                / f"{instance_name}__seq-{policy.sequencing_rule}"
                f"__route-{policy.routing_rule}__dispatch-{policy.dispatching_rule}.csv"
            )
            decision_trace_path = (
                results_dir
                / f"{instance_name}__seq-{policy.sequencing_rule}"
                f"__route-{policy.routing_rule}__dispatch-{policy.dispatching_rule}__decision_trace.jsonl"
            )
            run_id: int | None = None
            try:
                if decision_trace_path.exists():
                    decision_trace_path.unlink()

                if run_source == "ortools":
                    agent = OptimalReplaySchedulingAgent(
                        problem_data=problem_data,
                        decision_log_path=str(decision_trace_path),
                        time_limit_sec=ortools_max_time_sec,
                        require_optimal=ortools_require_optimal,
                    )
                else:
                    agent = HeuristicSchedulingAgent(decision_log_path=str(decision_trace_path))
                monitor = run_simulation_stepwise(
                    problem_data=problem_data,
                    event_log_path=str(log_path),
                    sequencing_rule=policy.sequencing_rule,
                    routing_rule=policy.routing_rule,
                    dispatching_rule=policy.dispatching_rule,
                    significant_digits=significant_digits,
                    agent=agent,
                )
                monitor.make_event_tracer()
                monitor.save_event_tracer()

                event_df = monitor.event_tracer.copy()
                decision_rows = _load_decision_trace(decision_trace_path)
                # Derive structured metrics from event log once, then persist.
                op_df = _extract_operation_metrics(event_df)
                job_df = _extract_job_completion(event_df)
                makespan = _compute_makespan(event_df, job_df)
                machine_df = _extract_machine_metrics(op_df, makespan)

                run_id = _insert_run_row(
                    conn=conn,
                    instance_id=instance_id,
                    policy=policy,
                    teacher_tier=teacher_tier,
                    seed=seed,
                    status="success",
                    makespan=makespan,
                    total_jobs=len(problem_data.get("job_info", {})),
                    completed_jobs=len(job_df),
                    event_log_path=log_path,
                    error_message=None,
                )
                _insert_metrics(
                    conn=conn,
                    run_id=run_id,
                    instance_name=instance_name,
                    policy=policy,
                    makespan=makespan,
                    operation_df=op_df,
                    machine_df=machine_df,
                    job_completion_df=job_df,
                )
                _insert_log_documents(
                    conn=conn,
                    run_id=run_id,
                    instance_name=instance_name,
                    policy=policy,
                    makespan=makespan,
                    event_df=event_df,
                    chunk_size=log_chunk_size,
                )
                _insert_decision_traces(
                    conn=conn,
                    run_id=run_id,
                    decision_rows=decision_rows,
                )
                _insert_decision_memories(
                    conn=conn,
                    run_id=run_id,
                    instance_name=instance_name,
                    policy=policy,
                    makespan=makespan,
                    problem_data=problem_data,
                    operation_df=op_df,
                    machine_df=machine_df,
                    job_completion_df=job_df,
                )
                _insert_decision_cases(
                    conn=conn,
                    run_id=run_id,
                    instance_name=instance_name,
                    policy=policy,
                    teacher_tier=teacher_tier,
                    makespan=makespan,
                    decision_rows=decision_rows,
                    operation_df=op_df,
                    machine_df=machine_df,
                )
                _insert_instance_features(
                    conn=conn,
                    run_id=run_id,
                    instance_name=instance_name,
                    problem_data=problem_data,
                    operation_df=op_df,
                    machine_df=machine_df,
                )
                conn.commit()
                summary["success"] += 1
                print(
                    f"[OK] {instance_name} | seq={policy.sequencing_rule}, "
                    f"route={policy.routing_rule}, dispatch={policy.dispatching_rule}, "
                    f"makespan={makespan:.3f}"
                )
            except Exception as exc:
                conn.rollback()
                run_id = _insert_run_row(
                    conn=conn,
                    instance_id=instance_id,
                    policy=policy,
                    teacher_tier=teacher_tier,
                    seed=seed,
                    status="failed",
                    makespan=None,
                    total_jobs=len(problem_data.get("job_info", {})),
                    completed_jobs=None,
                    event_log_path=log_path,
                    error_message=str(exc),
                )
                conn.commit()
                summary["failed"] += 1
                print(
                    f"[FAIL] {instance_name} | seq={policy.sequencing_rule}, "
                    f"route={policy.routing_rule}, dispatch={policy.dispatching_rule}, "
                    f"run_id={run_id}, error={exc}"
                )

        _refresh_teacher_tiers_for_instance(
            conn=conn,
            instance_id=instance_id,
            mode=teacher_mode,
            teacher_dispatching_rules=teacher_dispatching_rules,
            teacher_top_k=teacher_top_k,
            teacher_source=teacher_source,
        )
        conn.commit()

    conn.close()
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    # Keep CLI explicit so this module can be used both as script and importable library.
    parser = argparse.ArgumentParser(
        description="Run JSSP simulation with multiple heuristics and store results in SQLite for RAG."
    )
    parser.add_argument("--dt-root", type=Path, default=Path("DT"))
    parser.add_argument("--db-path", type=Path, default=Path("jssp_rag.db"))
    parser.add_argument("--results-dir", type=Path, default=Path("DT") / "results" / "JSSP_RAG")
    parser.add_argument("--instances", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sequencing-rules", type=str, default=",".join(DEFAULT_SEQUENCING_RULES))
    parser.add_argument("--dispatching-rules", type=str, default=",".join(DEFAULT_DISPATCHING_RULES))
    parser.add_argument("--routing-rule", type=str, default=DEFAULT_ROUTING_RULE)
    parser.add_argument(
        "--teacher-source",
        choices=["heuristic", "ortools"],
        default=DEFAULT_TEACHER_SOURCE,
        help="Teacher trace source. heuristic uses selected dispatching rules; ortools replays a solver schedule.",
    )
    parser.add_argument(
        "--teacher-dispatching-rules",
        type=str,
        default=",".join(DEFAULT_TEACHER_DISPATCHING_RULES),
        help="Dispatching rules treated as teacher-quality corpus entries. Default: SPT.",
    )
    parser.add_argument(
        "--teacher-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, only teacher dispatching-rule runs are inserted into the DB.",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=["dispatch_rule", "best_run"],
        default="dispatch_rule",
        help="How teacher_tier is assigned. dispatch_rule: based on --teacher-dispatching-rules. best_run: best makespan runs per instance.",
    )
    parser.add_argument(
        "--teacher-top-k",
        type=int,
        default=1,
        help="When --teacher-mode best_run is used, mark the top-k successful runs per instance as teacher.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ortools-max-time-sec",
        type=float,
        default=60.0,
        help="Time limit for OR-Tools teacher solve. Ignored unless --teacher-source ortools.",
    )
    parser.add_argument(
        "--ortools-require-optimal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require OR-Tools to prove optimality instead of accepting the best feasible schedule.",
    )
    parser.add_argument("--significant-digits", type=int, default=10)
    parser.add_argument("--log-chunk-size", type=int, default=DEFAULT_LOG_CHUNK_SIZE)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_jssp_dir = args.dt_root / "data" / "raw" / "JSSP"
    selected_instances = _split_csv_arg(args.instances)
    if not selected_instances:
        selected_instances = _list_jssp_instances(raw_jssp_dir)

    if args.limit and args.limit > 0:
        selected_instances = selected_instances[: args.limit]

    if not selected_instances:
        raise SystemExit("No JSSP instances found. Check DT/data/raw/JSSP or --instances option.")

    sequencing_rules = _split_csv_arg(args.sequencing_rules) or list(DEFAULT_SEQUENCING_RULES)
    dispatching_rules = _split_csv_arg(args.dispatching_rules) or list(DEFAULT_DISPATCHING_RULES)
    teacher_dispatching_rules = set(
        _split_csv_arg(args.teacher_dispatching_rules) or list(DEFAULT_TEACHER_DISPATCHING_RULES)
    )
    if args.teacher_only:
        if args.teacher_source == "heuristic":
            dispatching_rules = [rule for rule in dispatching_rules if rule in teacher_dispatching_rules]
            if not dispatching_rules:
                raise SystemExit("No dispatching rules remain after applying --teacher-only.")
        else:
            dispatching_rules = []
            sequencing_rules = []
    policies = [
        HeuristicPolicy(seq, args.routing_rule, disp)
        for seq in sequencing_rules
        for disp in dispatching_rules
    ]
    total_runs = len(selected_instances) * (len(policies) + (1 if args.teacher_source == "ortools" else 0))

    print(
        f"Start batch: instances={len(selected_instances)}, policies={len(policies)}, "
        f"teacher_source={args.teacher_source}, total_runs={total_runs}"
    )
    summary = run_jssp_batch_to_db(
        dt_root=args.dt_root,
        db_path=args.db_path,
        results_dir=args.results_dir,
        instance_names=selected_instances,
        policies=policies,
        teacher_dispatching_rules=teacher_dispatching_rules,
        teacher_mode=args.teacher_mode,
        teacher_top_k=args.teacher_top_k,
        teacher_source=args.teacher_source,
        ortools_max_time_sec=args.ortools_max_time_sec,
        ortools_require_optimal=args.ortools_require_optimal,
        seed=args.seed,
        significant_digits=args.significant_digits,
        force_convert=args.force_convert,
        overwrite=args.overwrite,
        log_chunk_size=args.log_chunk_size,
    )
    print(
        f"Done. success={summary['success']}, failed={summary['failed']}, "
        f"skipped={summary['skipped']}, db={args.db_path}"
    )


if __name__ == "__main__":
    main()
