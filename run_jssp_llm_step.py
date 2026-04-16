import argparse
import os
from pathlib import Path

from DT.decision_policy import OpenAIDecisionPolicy
from DT.modeling import load_data_from_unified_csv, run_simulation
from DT.utils.benchmarking_converter import convert_benchmarking_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JSSP simulation with step-by-step LLM decisions at routing/dispatch/sequencing points."
    )
    parser.add_argument("--dt-root", type=Path, default=Path("DT"))
    parser.add_argument("--instance", type=str, default="la01")
    parser.add_argument("--sequencing-rule", type=str, default="FIFO")
    parser.add_argument("--routing-rule", type=str, default="FIFO")
    parser.add_argument("--dispatching-rule", type=str, default="FIFO")
    parser.add_argument("--event-log-path", type=Path, default=Path("DT") / "results" / "JSSP_LLM" / "la01_llm.csv")
    parser.add_argument("--significant-digits", type=int, default=10)

    parser.add_argument("--api-url", type=str, default=os.getenv("LLM_API_URL", ""))
    parser.add_argument("--api-key", type=str, default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--timeout-sec", type=int, default=30)

    parser.add_argument("--rag-db-path", type=Path, default=Path("jssp_rag.db"))
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument(
        "--use-rag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable RAG log-chunk retrieval for each LLM decision.",
    )
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_url:
        raise SystemExit("Set --api-url or LLM_API_URL.")
    if not args.api_key:
        raise SystemExit("Set --api-key or LLM_API_KEY.")

    dt_root = args.dt_root
    preprocessed_dir = dt_root / "data" / "preprocessed"
    raw_jssp_dir = dt_root / "data" / "raw" / "JSSP"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    args.event_log_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessed_csv = preprocessed_dir / f"problem_{args.instance}.csv"
    raw_txt = raw_jssp_dir / f"{args.instance}.txt"
    if args.force_convert or not preprocessed_csv.exists():
        if not raw_txt.exists():
            raise SystemExit(f"Cannot find source txt: {raw_txt}")
        convert_benchmarking_data(str(raw_txt), str(preprocessed_dir), "JSSP")

    data = load_data_from_unified_csv(str(preprocessed_dir), args.instance)
    if not data:
        raise SystemExit(f"Failed to load instance: {args.instance}")

    policy = OpenAIDecisionPolicy(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        rag_db_path=str(args.rag_db_path) if args.rag_db_path else None,
        rag_top_k=args.rag_top_k,
        use_rag=args.use_rag,
    )

    monitor = run_simulation(
        problem_data=data,
        event_log_path=str(args.event_log_path),
        sequencing_rule=args.sequencing_rule,
        routing_rule=args.routing_rule,
        dispatching_rule=args.dispatching_rule,
        significant_digits=args.significant_digits,
        decision_policy=policy,
    )
    monitor.make_event_tracer()
    monitor.save_event_tracer()
    print(f"Saved event log: {args.event_log_path}")


if __name__ == "__main__":
    main()
