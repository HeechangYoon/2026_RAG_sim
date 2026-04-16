import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two JSSP event logs.")
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def load_event_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing event log: {path}")
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    df["event"] = df["event"].astype(str).str.lower()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df


def completion_by_job(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["event"].isin(["job transferred to sink", "job completed"])].copy()
    if completed.empty:
        return pd.DataFrame(columns=["part_id", "completion_time"])
    result = (
        completed.groupby("part_id", as_index=False)["time"]
        .max()
        .rename(columns={"time": "completion_time"})
    )
    return result


def operation_finish(df: pd.DataFrame) -> pd.DataFrame:
    ops = df[df["event"].eq("operation complete")].copy()
    if ops.empty:
        return pd.DataFrame(columns=["part_id", "operation", "machine", "finish_time"])
    return ops.rename(columns={"time": "finish_time"})[["part_id", "operation", "machine", "finish_time"]]


def print_summary(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame, top_n: int) -> None:
    baseline_completion = completion_by_job(baseline_df)
    candidate_completion = completion_by_job(candidate_df)

    baseline_makespan = baseline_completion["completion_time"].max() if not baseline_completion.empty else float("nan")
    candidate_makespan = candidate_completion["completion_time"].max() if not candidate_completion.empty else float("nan")

    print(f"baseline_makespan={baseline_makespan}")
    print(f"candidate_makespan={candidate_makespan}")
    print(f"delta_makespan={candidate_makespan - baseline_makespan}")

    merged_completion = baseline_completion.merge(
        candidate_completion,
        on="part_id",
        how="outer",
        suffixes=("_baseline", "_candidate"),
    )
    if not merged_completion.empty:
        merged_completion["delta_completion"] = (
            merged_completion["completion_time_candidate"] - merged_completion["completion_time_baseline"]
        )
        worst_jobs = merged_completion.sort_values("delta_completion", ascending=False).head(top_n)
        print("\nWorst job completion regressions:")
        for row in worst_jobs.itertuples(index=False):
            print(
                f"- {row.part_id}: baseline={row.completion_time_baseline}, "
                f"candidate={row.completion_time_candidate}, delta={row.delta_completion}"
            )

    baseline_ops = operation_finish(baseline_df)
    candidate_ops = operation_finish(candidate_df)
    merged_ops = baseline_ops.merge(
        candidate_ops,
        on=["part_id", "operation", "machine"],
        how="outer",
        suffixes=("_baseline", "_candidate"),
    )
    if not merged_ops.empty:
        merged_ops["delta_finish"] = merged_ops["finish_time_candidate"] - merged_ops["finish_time_baseline"]
        worst_ops = merged_ops.sort_values("delta_finish", ascending=False).head(top_n)
        print("\nWorst operation finish regressions:")
        for row in worst_ops.itertuples(index=False):
            print(
                f"- {row.part_id}/{row.operation}/{row.machine}: "
                f"baseline={row.finish_time_baseline}, candidate={row.finish_time_candidate}, delta={row.delta_finish}"
            )


def main() -> None:
    args = parse_args()
    baseline_df = load_event_log(args.baseline_log)
    candidate_df = load_event_log(args.candidate_log)
    print_summary(baseline_df, candidate_df, args.top_n)


if __name__ == "__main__":
    main()
