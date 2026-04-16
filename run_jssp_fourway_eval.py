import argparse
import json
import os
from pathlib import Path

import pandas as pd

from DT.agents import AgentDecisionError
from DT.agents import OpenAISchedulingAgent, SelectiveDecisionAgent
from DT.modeling import load_data_from_unified_csv, run_simulation_stepwise
from DT.utils.benchmarking_converter import convert_benchmarking_data


def _resolve_raw_jssp_source(raw_jssp_dir: Path, instance: str) -> Path:
    candidates = [raw_jssp_dir / instance, raw_jssp_dir / f"{instance}.txt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for SPT vs LLM vs LLM+RAG(benchmark DB) vs LLM+RAG(synthetic DB) on JSSP."
    )
    parser.add_argument("--dt-root", type=Path, default=Path("DT"))
    parser.add_argument("--instances", type=str, default="")
    parser.add_argument("--instances-from-summary", type=Path, default=None)
    parser.add_argument(
        "--error-columns",
        type=str,
        default="llm_error,llm_rag_bench_error,llm_rag_synth_error",
        help="Comma-separated error columns to inspect when using --instances-from-summary.",
    )
    parser.add_argument(
        "--error-contains",
        type=str,
        default="",
        help="Only keep instances whose selected error columns contain this substring.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("DT") / "results" / "JSSP_FOURWAY_EVAL")
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("DT") / "results" / "JSSP_FOURWAY_EVAL" / "runs.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("DT") / "results" / "JSSP_FOURWAY_EVAL" / "summary.csv",
    )
    parser.add_argument("--sequencing-rule", type=str, default="FIFO")
    parser.add_argument("--routing-rule", type=str, default="FIFO")
    parser.add_argument("--dispatching-rule", type=str, default="SPT")
    parser.add_argument("--significant-digits", type=int, default=10)
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
    )
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "")))
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "gpt-5-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument(
        "--response-mode",
        choices=["verbose_reason", "short_reason", "choice_only"],
        default="verbose_reason",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print the exact prompt before each LLM decision call.",
    )
    parser.add_argument("--adaptive-max-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-token-step", type=int, default=200)
    parser.add_argument("--adaptive-token-cap", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--benchmark-rag-db-path", type=Path, required=True)
    parser.add_argument("--synthetic-rag-db-path", type=Path, required=True)
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument("--agent-decision-types", type=str, default="dispatch")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def list_jssp_instances(dt_root: Path) -> list[str]:
    raw_dir = dt_root / "data" / "raw" / "JSSP"
    if not raw_dir.exists():
        return []
    names = set()
    for path in raw_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix == ".txt":
            names.add(path.stem)
        elif path.suffix == "":
            names.add(path.name)
    return sorted(names)


def list_instances_from_summary(summary_csv: Path, error_columns: list[str], error_contains: str) -> list[str]:
    if not summary_csv.exists():
        raise SystemExit(f"Cannot find summary CSV: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if "instance" not in df.columns:
        raise SystemExit(f"Summary CSV does not contain 'instance': {summary_csv}")

    valid_cols = [col for col in error_columns if col in df.columns]
    if not valid_cols:
        raise SystemExit(f"None of the requested error columns exist in {summary_csv}: {error_columns}")

    mask = pd.Series(False, index=df.index)
    needle = (error_contains or "").strip()
    for col in valid_cols:
        values = df[col].fillna("").astype(str)
        col_mask = values.str.len().gt(0)
        if needle:
            col_mask = col_mask & values.str.contains(needle, regex=False)
        mask = mask | col_mask
    return df.loc[mask, "instance"].astype(str).drop_duplicates().tolist()


def load_problem(dt_root: Path, instance: str, force_convert: bool):
    preprocessed_dir = dt_root / "data" / "preprocessed"
    raw_jssp_dir = dt_root / "data" / "raw" / "JSSP"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_csv = preprocessed_dir / f"problem_{instance}.csv"
    raw_txt = _resolve_raw_jssp_source(raw_jssp_dir, instance)
    if force_convert or not preprocessed_csv.exists():
        if not raw_txt.exists():
            raise SystemExit(f"Cannot find source JSSP file: {raw_txt}")
        convert_benchmarking_data(str(raw_txt), str(preprocessed_dir), "JSSP")
    data = load_data_from_unified_csv(str(preprocessed_dir), instance)
    if not data:
        raise SystemExit(f"Failed to load instance: {instance}")
    return data


def completion_by_job(df: pd.DataFrame) -> dict[str, float]:
    completed = df[df["event"].isin(["job transferred to sink", "job completed"])].copy()
    if completed.empty:
        return {}
    grouped = completed.groupby("part_id", as_index=False)["time"].max()
    return {str(row.part_id): float(row.time) for row in grouped.itertuples(index=False)}


def makespan_from_monitor(monitor) -> float:
    event_df = monitor.event_tracer.copy()
    event_df.columns = [str(col).strip().lower() for col in event_df.columns]
    event_df["event"] = event_df["event"].astype(str).str.lower()
    event_df["time"] = pd.to_numeric(event_df["time"], errors="coerce")
    comp = completion_by_job(event_df)
    return max(comp.values()) if comp else float("nan")


def token_usage_from_decision_log(path: Path) -> dict[str, float]:
    usage = {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "usage_records": 0.0,
    }
    if not path.exists():
        return usage
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            item = row.get("usage")
            if not isinstance(item, dict):
                continue
            usage["prompt_tokens"] += float(item.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] += float(item.get("completion_tokens", 0) or 0)
            usage["total_tokens"] += float(item.get("total_tokens", 0) or 0)
            usage["usage_records"] += 1.0
    return usage


def build_llm_agent(
    args: argparse.Namespace,
    use_rag: bool,
    rag_db_path: Path | None,
    decision_log_path: Path,
    prompt_log_path: Path,
):
    base_agent = OpenAISchedulingAgent(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        rag_db_path=str(rag_db_path) if rag_db_path else None,
        rag_top_k=args.rag_top_k,
        use_rag=use_rag,
        decision_log_path=str(decision_log_path),
        prompt_log_path=str(prompt_log_path),
        show_prompt=args.show_prompts,
        fail_on_invalid_action=True,
        adaptive_max_tokens=args.adaptive_max_tokens,
        adaptive_token_step=args.adaptive_token_step,
        adaptive_token_cap=args.adaptive_token_cap,
        response_mode=args.response_mode,
    )
    decision_types = {item.strip().lower() for item in args.agent_decision_types.split(",") if item.strip()}
    if not decision_types or "all" in decision_types:
        return base_agent
    return SelectiveDecisionAgent(agent=base_agent, decision_types=decision_types)


def run_case(
    problem_data,
    args: argparse.Namespace,
    mode: str,
    event_log_path: Path,
    decision_log_path: Path,
    prompt_log_path: Path,
):
    if decision_log_path.exists():
        decision_log_path.unlink()
    if prompt_log_path.exists():
        prompt_log_path.unlink()

    if mode == "spt":
        from DT.agents import HeuristicSchedulingAgent

        agent = HeuristicSchedulingAgent(decision_log_path=str(decision_log_path))
    elif mode == "llm":
        agent = build_llm_agent(
            args,
            use_rag=False,
            rag_db_path=None,
            decision_log_path=decision_log_path,
            prompt_log_path=prompt_log_path,
        )
    elif mode == "llm_rag_bench":
        agent = build_llm_agent(
            args,
            use_rag=True,
            rag_db_path=args.benchmark_rag_db_path,
            decision_log_path=decision_log_path,
            prompt_log_path=prompt_log_path,
        )
    elif mode == "llm_rag_synth":
        agent = build_llm_agent(
            args,
            use_rag=True,
            rag_db_path=args.synthetic_rag_db_path,
            decision_log_path=decision_log_path,
            prompt_log_path=prompt_log_path,
        )
    else:
        raise ValueError(mode)

    monitor = run_simulation_stepwise(
        problem_data=problem_data,
        event_log_path=str(event_log_path),
        sequencing_rule=args.sequencing_rule,
        routing_rule=args.routing_rule,
        dispatching_rule=args.dispatching_rule,
        significant_digits=args.significant_digits,
        agent=agent,
    )
    monitor.make_event_tracer()
    monitor.save_event_tracer()
    return makespan_from_monitor(monitor)


def safe_run_case(
    problem_data,
    args: argparse.Namespace,
    mode: str,
    event_log_path: Path,
    decision_log_path: Path,
    prompt_log_path: Path,
):
    try:
        return run_case(problem_data, args, mode, event_log_path, decision_log_path, prompt_log_path), ""
    except AgentDecisionError as exc:
        if not args.continue_on_error:
            raise
        return float("nan"), f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        if not args.continue_on_error:
            raise
        return float("nan"), f"{type(exc).__name__}: {exc}"


def safe_delta(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    return a - b


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    method_cols = [
        "spt_makespan",
        "llm_makespan",
        "llm_rag_bench_makespan",
        "llm_rag_synth_makespan",
        "delta_llm_vs_spt",
        "delta_bench_vs_spt",
        "delta_synth_vs_spt",
        "delta_bench_vs_llm",
        "delta_synth_vs_llm",
        "delta_synth_vs_bench",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
        "llm_usage_records",
        "llm_rag_bench_prompt_tokens",
        "llm_rag_bench_completion_tokens",
        "llm_rag_bench_total_tokens",
        "llm_rag_bench_usage_records",
        "llm_rag_synth_prompt_tokens",
        "llm_rag_synth_completion_tokens",
        "llm_rag_synth_total_tokens",
        "llm_rag_synth_usage_records",
    ]
    error_cols = ["spt_error", "llm_error", "llm_rag_bench_error", "llm_rag_synth_error"]

    rows = []
    for instance, g in df.groupby("instance", sort=True):
        row = {
            "instance": instance,
            "num_repeats": int(len(g)),
        }
        for col in method_cols:
            values = pd.to_numeric(g[col], errors="coerce")
            valid = values.dropna()
            row[f"{col}_count"] = int(valid.shape[0])
            row[f"{col}_mean"] = float(valid.mean()) if not valid.empty else float("nan")
            row[f"{col}_std"] = float(valid.std(ddof=0)) if not valid.empty else float("nan")
            row[f"{col}_min"] = float(valid.min()) if not valid.empty else float("nan")
            row[f"{col}_max"] = float(valid.max()) if not valid.empty else float("nan")

        for col in error_cols:
            errs = g[col].fillna("").astype(str)
            nonempty = [e for e in errs if e]
            row[f"{col}_count"] = len(nonempty)
            row[f"{col}_sample"] = nonempty[0] if nonempty else ""

        # win-rate style summary fields that are handy for later plots/tables.
        bench_vs_llm = pd.to_numeric(g["delta_bench_vs_llm"], errors="coerce").dropna()
        synth_vs_llm = pd.to_numeric(g["delta_synth_vs_llm"], errors="coerce").dropna()
        synth_vs_bench = pd.to_numeric(g["delta_synth_vs_bench"], errors="coerce").dropna()
        row["bench_better_than_llm_rate"] = float((bench_vs_llm < 0).mean()) if not bench_vs_llm.empty else float("nan")
        row["synth_better_than_llm_rate"] = float((synth_vs_llm < 0).mean()) if not synth_vs_llm.empty else float("nan")
        row["synth_better_than_bench_rate"] = float((synth_vs_bench < 0).mean()) if not synth_vs_bench.empty else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or LLM_API_KEY before running batch LLM evaluation.")

    instances = split_csv_arg(args.instances)
    if not instances:
        if args.instances_from_summary:
            instances = list_instances_from_summary(
                args.instances_from_summary,
                split_csv_arg(args.error_columns),
                args.error_contains,
            )
        else:
            instances = list_jssp_instances(args.dt_root)
    if args.limit and args.limit > 0:
        instances = instances[: args.limit]
    if not instances:
        raise SystemExit("No JSSP instances found. Use --instances or check DT/data/raw/JSSP.")
    if args.num_repeats < 1:
        raise SystemExit("--num-repeats must be >= 1")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.runs_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for instance in instances:
        problem_data = load_problem(args.dt_root, instance, args.force_convert)
        print(f"\n[INSTANCE] {instance}")
        for repeat_idx in range(1, args.num_repeats + 1):
            suffix = f"{instance}_r{repeat_idx:02d}"
            print(f"  [REPEAT {repeat_idx}/{args.num_repeats}]")
            spt_log = args.results_dir / f"{suffix}_spt.csv"
            spt_decision = args.results_dir / f"{suffix}_spt_decision.jsonl"
            spt_prompt = args.results_dir / f"{suffix}_spt_prompt.jsonl"
            llm_log = args.results_dir / f"{suffix}_llm.csv"
            llm_decision = args.results_dir / f"{suffix}_llm_decision.jsonl"
            llm_prompt = args.results_dir / f"{suffix}_llm_prompt.jsonl"
            rag_bench_log = args.results_dir / f"{suffix}_llm_rag_bench.csv"
            rag_bench_decision = args.results_dir / f"{suffix}_llm_rag_bench_decision.jsonl"
            rag_bench_prompt = args.results_dir / f"{suffix}_llm_rag_bench_prompt.jsonl"
            rag_synth_log = args.results_dir / f"{suffix}_llm_rag_synth.csv"
            rag_synth_decision = args.results_dir / f"{suffix}_llm_rag_synth_decision.jsonl"
            rag_synth_prompt = args.results_dir / f"{suffix}_llm_rag_synth_prompt.jsonl"

            spt_makespan, spt_error = safe_run_case(problem_data, args, "spt", spt_log, spt_decision, spt_prompt)
            llm_makespan, llm_error = safe_run_case(problem_data, args, "llm", llm_log, llm_decision, llm_prompt)
            rag_bench_makespan, rag_bench_error = safe_run_case(
                problem_data, args, "llm_rag_bench", rag_bench_log, rag_bench_decision, rag_bench_prompt
            )
            rag_synth_makespan, rag_synth_error = safe_run_case(
                problem_data, args, "llm_rag_synth", rag_synth_log, rag_synth_decision, rag_synth_prompt
            )
            llm_usage = token_usage_from_decision_log(llm_decision)
            rag_bench_usage = token_usage_from_decision_log(rag_bench_decision)
            rag_synth_usage = token_usage_from_decision_log(rag_synth_decision)

            row = {
                "instance": instance,
                "repeat_index": repeat_idx,
                "spt_makespan": spt_makespan,
                "llm_makespan": llm_makespan,
                "llm_rag_bench_makespan": rag_bench_makespan,
                "llm_rag_synth_makespan": rag_synth_makespan,
                "delta_llm_vs_spt": safe_delta(llm_makespan, spt_makespan),
                "delta_bench_vs_spt": safe_delta(rag_bench_makespan, spt_makespan),
                "delta_synth_vs_spt": safe_delta(rag_synth_makespan, spt_makespan),
                "delta_bench_vs_llm": safe_delta(rag_bench_makespan, llm_makespan),
                "delta_synth_vs_llm": safe_delta(rag_synth_makespan, llm_makespan),
                "delta_synth_vs_bench": safe_delta(rag_synth_makespan, rag_bench_makespan),
                "llm_prompt_tokens": llm_usage["prompt_tokens"],
                "llm_completion_tokens": llm_usage["completion_tokens"],
                "llm_total_tokens": llm_usage["total_tokens"],
                "llm_usage_records": llm_usage["usage_records"],
                "llm_rag_bench_prompt_tokens": rag_bench_usage["prompt_tokens"],
                "llm_rag_bench_completion_tokens": rag_bench_usage["completion_tokens"],
                "llm_rag_bench_total_tokens": rag_bench_usage["total_tokens"],
                "llm_rag_bench_usage_records": rag_bench_usage["usage_records"],
                "llm_rag_synth_prompt_tokens": rag_synth_usage["prompt_tokens"],
                "llm_rag_synth_completion_tokens": rag_synth_usage["completion_tokens"],
                "llm_rag_synth_total_tokens": rag_synth_usage["total_tokens"],
                "llm_rag_synth_usage_records": rag_synth_usage["usage_records"],
                "spt_error": spt_error,
                "llm_error": llm_error,
                "llm_rag_bench_error": rag_bench_error,
                "llm_rag_synth_error": rag_synth_error,
                "spt_log": str(spt_log),
                "llm_log": str(llm_log),
                "llm_rag_bench_log": str(rag_bench_log),
                "llm_rag_synth_log": str(rag_synth_log),
                "llm_decision_log": str(llm_decision),
                "llm_rag_bench_decision_log": str(rag_bench_decision),
                "llm_rag_synth_decision_log": str(rag_synth_decision),
            }
            rows.append(row)
            print(
                f"    SPT={spt_makespan:.3f} | "
                f"LLM={llm_makespan:.3f} | "
                f"RAG(bench)={rag_bench_makespan:.3f} | "
                f"RAG(synth)={rag_synth_makespan:.3f}"
            )
            print(
                f"      tokens LLM={llm_usage['total_tokens']:.0f} | "
                f"bench={rag_bench_usage['total_tokens']:.0f} | "
                f"synth={rag_synth_usage['total_tokens']:.0f}"
            )
            if llm_error:
                print(f"      LLM error: {llm_error}")
            if rag_bench_error:
                print(f"      RAG(bench) error: {rag_bench_error}")
            if rag_synth_error:
                print(f"      RAG(synth) error: {rag_synth_error}")

            runs_df = pd.DataFrame(rows)
            runs_df.to_csv(args.runs_csv, index=False)
            aggregate_runs(runs_df).to_csv(args.summary_csv, index=False)

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(args.runs_csv, index=False)
    summary_df = aggregate_runs(runs_df)
    summary_df.to_csv(args.summary_csv, index=False)
    print(f"\nSaved run-level results: {args.runs_csv}")
    print(f"Saved aggregated summary: {args.summary_csv}")
    if not summary_df.empty:
        print("\nAggregate means:")
        print(
            summary_df[
                [
                    "delta_llm_vs_spt_mean",
                    "delta_bench_vs_spt_mean",
                    "delta_synth_vs_spt_mean",
                    "delta_bench_vs_llm_mean",
                    "delta_synth_vs_llm_mean",
                    "delta_synth_vs_bench_mean",
                ]
            ].mean(numeric_only=True)
        )


if __name__ == "__main__":
    main()
