import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simple human-readable summary from run_jssp_fourway_eval aggregated results."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--mean-instance-summary-csv", type=Path, default=None)
    parser.add_argument("--min-instance-summary-csv", type=Path, default=None)
    return parser.parse_args()


def choose_best_method(row: pd.Series, suffix: str) -> str:
    candidates = {
        "SPT": row.get(f"spt_makespan_{suffix}"),
        "LLM": row.get(f"llm_makespan_{suffix}"),
        "RAG_BENCH": row.get(f"llm_rag_bench_makespan_{suffix}"),
        "RAG_SYNTH": row.get(f"llm_rag_synth_makespan_{suffix}"),
    }
    numeric = {k: float(v) for k, v in candidates.items() if pd.notna(v)}
    if not numeric:
        return ""
    best_value = min(numeric.values())
    winners = sorted(k for k, v in numeric.items() if v == best_value)
    return "|".join(winners)


def build_mean_instance_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "instance": df["instance"],
            "num_repeats": df.get("num_repeats"),
            "spt_mean": df.get("spt_makespan_mean"),
            "llm_mean": df.get("llm_makespan_mean"),
            "rag_bench_mean": df.get("llm_rag_bench_makespan_mean"),
            "rag_synth_mean": df.get("llm_rag_synth_makespan_mean"),
            "llm_vs_spt_mean": df.get("delta_llm_vs_spt_mean"),
            "bench_vs_spt_mean": df.get("delta_bench_vs_spt_mean"),
            "synth_vs_spt_mean": df.get("delta_synth_vs_spt_mean"),
            "bench_vs_llm_mean": df.get("delta_bench_vs_llm_mean"),
            "synth_vs_llm_mean": df.get("delta_synth_vs_llm_mean"),
            "synth_vs_bench_mean": df.get("delta_synth_vs_bench_mean"),
            "llm_std": df.get("llm_makespan_std"),
            "rag_bench_std": df.get("llm_rag_bench_makespan_std"),
            "rag_synth_std": df.get("llm_rag_synth_makespan_std"),
            "llm_total_tokens_mean": df.get("llm_total_tokens_mean"),
            "rag_bench_total_tokens_mean": df.get("llm_rag_bench_total_tokens_mean"),
            "rag_synth_total_tokens_mean": df.get("llm_rag_synth_total_tokens_mean"),
            "llm_total_tokens_std": df.get("llm_total_tokens_std"),
            "rag_bench_total_tokens_std": df.get("llm_rag_bench_total_tokens_std"),
            "rag_synth_total_tokens_std": df.get("llm_rag_synth_total_tokens_std"),
            "llm_error_count": df.get("llm_error_count"),
            "rag_bench_error_count": df.get("llm_rag_bench_error_count"),
            "rag_synth_error_count": df.get("llm_rag_synth_error_count"),
        }
    )
    out["best_method_mean"] = df.apply(lambda row: choose_best_method(row, "mean"), axis=1)
    return out


def build_min_instance_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "instance": df["instance"],
            "num_repeats": df.get("num_repeats"),
            "spt_min": df.get("spt_makespan_min"),
            "llm_min": df.get("llm_makespan_min"),
            "rag_bench_min": df.get("llm_rag_bench_makespan_min"),
            "rag_synth_min": df.get("llm_rag_synth_makespan_min"),
            "llm_vs_spt_min": df.get("delta_llm_vs_spt_min"),
            "bench_vs_spt_min": df.get("delta_bench_vs_spt_min"),
            "synth_vs_spt_min": df.get("delta_synth_vs_spt_min"),
            "bench_vs_llm_min": df.get("delta_bench_vs_llm_min"),
            "synth_vs_llm_min": df.get("delta_synth_vs_llm_min"),
            "synth_vs_bench_min": df.get("delta_synth_vs_bench_min"),
            "llm_total_tokens_min": df.get("llm_total_tokens_min"),
            "rag_bench_total_tokens_min": df.get("llm_rag_bench_total_tokens_min"),
            "rag_synth_total_tokens_min": df.get("llm_rag_synth_total_tokens_min"),
            "llm_error_count": df.get("llm_error_count"),
            "rag_bench_error_count": df.get("llm_rag_bench_error_count"),
            "rag_synth_error_count": df.get("llm_rag_synth_error_count"),
        }
    )
    out["best_method_min"] = df.apply(lambda row: choose_best_method(row, "min"), axis=1)
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise SystemExit(f"Summary CSV is empty: {args.summary_csv}")

    mean_instance_summary = build_mean_instance_summary(df)
    min_instance_summary = build_min_instance_summary(df)

    mean_path = args.mean_instance_summary_csv or args.summary_csv.with_name("mean_instance_summary.csv")
    min_path = args.min_instance_summary_csv or args.summary_csv.with_name("min_instance_summary.csv")
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    min_path.parent.mkdir(parents=True, exist_ok=True)

    mean_instance_summary.to_csv(mean_path, index=False)
    min_instance_summary.to_csv(min_path, index=False)

    print(f"Saved mean instance summary: {mean_path}")
    print(f"Saved min instance summary: {min_path}")
    print()
    print("Mean instance summary preview:")
    print(mean_instance_summary.head().to_string(index=False))
    print()
    print("Min instance summary preview:")
    print(min_instance_summary.head().to_string(index=False))


if __name__ == "__main__":
    main()
