import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze changed dispatch decisions against baseline/candidate logs.")
    parser.add_argument("--decision-log-path", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--decision-type", type=str, default="dispatch")
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing decision log: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as fp:
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


def load_event_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing event log: {path}")
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    df["event"] = df["event"].astype(str).str.lower()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df


def completion_by_job(df: pd.DataFrame) -> dict[str, float]:
    completed = df[df["event"].isin(["job transferred to sink", "job completed"])].copy()
    if completed.empty:
        return {}
    grouped = completed.groupby("part_id", as_index=False)["time"].max()
    return {str(row.part_id): float(row.time) for row in grouped.itertuples(index=False)}


def makespan(df: pd.DataFrame) -> float:
    comp = completion_by_job(df)
    return max(comp.values()) if comp else float("nan")


def main() -> None:
    args = parse_args()
    records = [
        row for row in load_jsonl(args.decision_log_path)
        if row.get("decision_type") == args.decision_type and not row.get("error")
    ]
    changed = [row for row in records if row.get("parsed_choice") != row.get("default_action")]
    baseline_df = load_event_log(args.baseline_log)
    candidate_df = load_event_log(args.candidate_log)
    baseline_completion = completion_by_job(baseline_df)
    candidate_completion = completion_by_job(candidate_df)
    baseline_makespan = makespan(baseline_df)
    candidate_makespan = makespan(candidate_df)

    print(f"decision_type={args.decision_type}")
    print(f"changed_decisions={len(changed)}")
    print(f"baseline_makespan={baseline_makespan}")
    print(f"candidate_makespan={candidate_makespan}")
    print(f"delta_makespan={candidate_makespan - baseline_makespan}")

    print("\nChanged decision impact samples:")
    for row in changed[: args.top_n]:
        state = row.get("state", {}) if isinstance(row.get("state"), dict) else {}
        chosen = str(row.get("parsed_choice")) if row.get("parsed_choice") is not None else None
        default = str(row.get("default_action")) if row.get("default_action") is not None else None
        chosen_delta = None
        default_delta = None
        if chosen is not None:
            chosen_delta = candidate_completion.get(chosen, float("nan")) - baseline_completion.get(chosen, float("nan"))
        if default is not None:
            default_delta = candidate_completion.get(default, float("nan")) - baseline_completion.get(default, float("nan"))
        print(
            json.dumps(
                {
                    "sim_time": state.get("sim_time"),
                    "machine_type": state.get("machine_type"),
                    "candidates": row.get("candidates"),
                    "default_action": default,
                    "parsed_choice": chosen,
                    "reason": row.get("reason"),
                    "chosen_job_completion_delta": chosen_delta,
                    "default_job_completion_delta": default_delta,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
