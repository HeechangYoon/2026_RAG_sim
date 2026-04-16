import argparse
import os
from pathlib import Path

from DT.agents import (
    HFSchedulingAgent,
    HeuristicSchedulingAgent,
    OpenAISchedulingAgent,
    ReinforcementLearningAgent,
    SelectiveDecisionAgent,
)
from DT.modeling import load_data_from_unified_csv, run_simulation_stepwise
from DT.utils.benchmarking_converter import convert_benchmarking_data
from DT.utils.postprocessing import plot_gantt_chart


def _resolve_raw_jssp_source(raw_jssp_dir: Path, instance: str) -> Path:
    candidates = [raw_jssp_dir / instance, raw_jssp_dir / f"{instance}.txt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class DummyRLModel:
    """Example stub. Replace with a real RL model loader/inference wrapper."""

    def predict(self, decision_type, state, candidates):
        if not candidates:
            return None
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JSSP simulation with a generic agent invoked at each decision point."
    )
    parser.add_argument("--dt-root", type=Path, default=Path("DT"))
    parser.add_argument("--instance", type=str, default="la01")
    parser.add_argument("--sequencing-rule", type=str, default="FIFO")
    parser.add_argument("--routing-rule", type=str, default="FIFO")
    parser.add_argument("--dispatching-rule", type=str, default="FIFO")
    parser.add_argument("--event-log-path", type=Path, default=Path("DT") / "results" / "JSSP_AGENT" / "la01_agent.csv")
    parser.add_argument(
        "--save-gantt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate and save a gantt chart PNG from the event log after the run.",
    )
    parser.add_argument("--significant-digits", type=int, default=10)
    parser.add_argument("--agent-type", choices=["heuristic", "llm", "hf", "rl"], default="llm")
    parser.add_argument(
        "--agent-decision-types",
        type=str,
        default="all",
        help="Comma-separated decision types handled by the agent. Others use the simulator default. Use 'all' for every type.",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "")),
    )
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "gpt-5-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument(
        "--response-mode",
        choices=["verbose_reason", "short_reason", "choice_only"],
        default="verbose_reason",
        help="Control how much explanation the model returns.",
    )
    parser.add_argument("--adaptive-max-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-token-step", type=int, default=200)
    parser.add_argument("--adaptive-token-cap", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--hf-model-id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--hf-device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--hf-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--decision-log-path",
        type=Path,
        default=Path("DT") / "results" / "JSSP_AGENT" / "decision_trace.jsonl",
    )
    parser.add_argument(
        "--prompt-log-path",
        type=Path,
        default=None,
        help="Optional JSONL file to store the exact prompts sent to the agent.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print the exact prompt before each LLM/HF decision call.",
    )
    parser.add_argument(
        "--reset-log-files",
        action="store_true",
        help="Delete existing prompt/decision log files before starting the run.",
    )

    parser.add_argument("--rag-db-path", type=Path, default=Path("jssp_rag.db"))
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument(
        "--use-rag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable RAG log-chunk retrieval for LLM agent.",
    )
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def _load_problem(args: argparse.Namespace):
    dt_root = args.dt_root
    preprocessed_dir = dt_root / "data" / "preprocessed"
    raw_jssp_dir = dt_root / "data" / "raw" / "JSSP"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    args.event_log_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessed_csv = preprocessed_dir / f"problem_{args.instance}.csv"
    raw_txt = _resolve_raw_jssp_source(raw_jssp_dir, args.instance)
    if args.force_convert or not preprocessed_csv.exists():
        if not raw_txt.exists():
            raise SystemExit(f"Cannot find source JSSP file: {raw_txt}")
        convert_benchmarking_data(str(raw_txt), str(preprocessed_dir), "JSSP")

    data = load_data_from_unified_csv(str(preprocessed_dir), args.instance)
    if not data:
        raise SystemExit(f"Failed to load instance: {args.instance}")
    return data


def _build_agent(args: argparse.Namespace):
    if args.agent_type == "heuristic":
        return HeuristicSchedulingAgent()

    if args.agent_type == "llm":
        if not args.api_url:
            raise SystemExit("Set --api-url or LLM_API_URL for llm agent.")
        if not args.api_key:
            raise SystemExit("Set --api-key, OPENAI_API_KEY, or LLM_API_KEY for llm agent.")
        base_agent = OpenAISchedulingAgent(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            rag_db_path=str(args.rag_db_path) if args.rag_db_path else None,
            rag_top_k=args.rag_top_k,
            use_rag=args.use_rag,
            decision_log_path=str(args.decision_log_path) if args.decision_log_path else None,
            prompt_log_path=str(args.prompt_log_path) if args.prompt_log_path else None,
            show_prompt=args.show_prompts,
            fail_on_invalid_action=True,
            adaptive_max_tokens=args.adaptive_max_tokens,
            adaptive_token_step=args.adaptive_token_step,
            adaptive_token_cap=args.adaptive_token_cap,
            response_mode=args.response_mode,
        )
        return _wrap_agent_for_decision_types(base_agent, args.agent_decision_types)

    if args.agent_type == "hf":
        base_agent = HFSchedulingAgent(
            model_id=args.hf_model_id,
            temperature=args.temperature,
            max_new_tokens=args.hf_max_new_tokens,
            device=args.hf_device,
            rag_db_path=str(args.rag_db_path) if args.rag_db_path else None,
            rag_top_k=args.rag_top_k,
            use_rag=args.use_rag,
            decision_log_path=str(args.decision_log_path) if args.decision_log_path else None,
            prompt_log_path=str(args.prompt_log_path) if args.prompt_log_path else None,
            show_prompt=args.show_prompts,
            fail_on_invalid_action=True,
            response_mode=args.response_mode,
        )
        return _wrap_agent_for_decision_types(base_agent, args.agent_decision_types)

    # Placeholder path for RL experiments until a concrete model loader is plugged in.
    base_agent = ReinforcementLearningAgent(policy_model=DummyRLModel())
    return _wrap_agent_for_decision_types(base_agent, args.agent_decision_types)


def _wrap_agent_for_decision_types(agent, raw_value: str):
    text = (raw_value or "").strip().lower()
    if not text or text == "all":
        return agent
    decision_types = {item.strip().lower() for item in raw_value.split(",") if item.strip()}
    return SelectiveDecisionAgent(agent=agent, decision_types=decision_types)


def main() -> None:
    args = parse_args()
    if args.reset_log_files:
        for path in [args.prompt_log_path, args.decision_log_path]:
            if path and path.exists():
                path.unlink()
    data = _load_problem(args)
    agent = _build_agent(args)
    monitor = run_simulation_stepwise(
        problem_data=data,
        event_log_path=str(args.event_log_path),
        sequencing_rule=args.sequencing_rule,
        routing_rule=args.routing_rule,
        dispatching_rule=args.dispatching_rule,
        significant_digits=args.significant_digits,
        agent=agent,
    )
    monitor.make_event_tracer()
    monitor.save_event_tracer()
    print(f"Saved event log: {args.event_log_path}")
    if args.save_gantt:
        plot_gantt_chart(str(args.event_log_path))


if __name__ == "__main__":
    main()
