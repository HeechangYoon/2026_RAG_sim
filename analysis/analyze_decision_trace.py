import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze agent decision trace JSONL output.")
    parser.add_argument(
        "--decision-log-path",
        type=Path,
        default=Path("DT") / "results" / "JSSP_AGENT" / "decision_trace.jsonl",
    )
    parser.add_argument("--decision-type", type=str, default="dispatch")
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing decision log: {path}")
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def main() -> None:
    args = parse_args()
    records = [r for r in load_records(args.decision_log_path) if r.get("decision_type") == args.decision_type]
    if not records:
        print(f"No records found for decision_type={args.decision_type}")
        return

    total = len(records)
    errors = [r for r in records if r.get("error")]
    valid = [r for r in records if not r.get("error")]
    changed = [
        r
        for r in valid
        if r.get("parsed_choice") is not None and r.get("parsed_choice") != r.get("default_action")
    ]

    usage_records = [r.get("usage", {}) for r in valid if isinstance(r.get("usage"), dict)]
    total_prompt = sum(int(u.get("prompt_tokens", 0) or 0) for u in usage_records)
    total_completion = sum(int(u.get("completion_tokens", 0) or 0) for u in usage_records)
    total_tokens = sum(int(u.get("total_tokens", 0) or 0) for u in usage_records)

    print(f"decision_type={args.decision_type}")
    print(f"total_records={total}")
    print(f"valid_records={len(valid)}")
    print(f"error_records={len(errors)}")
    print(f"changed_vs_default={len(changed)} ({len(changed) / max(len(valid), 1):.1%} of valid)")
    if usage_records:
        print(
            f"tokens prompt={total_prompt} completion={total_completion} total={total_tokens} "
            f"avg_total={total_tokens / len(usage_records):.1f}"
        )

    error_counter = Counter(str(r.get("error")) for r in errors)
    if error_counter:
        print("\nTop errors:")
        for key, count in error_counter.most_common(args.top_n):
            print(f"- {key}: {count}")

    choice_counter = Counter(str(r.get("parsed_choice")) for r in valid if r.get("parsed_choice"))
    if choice_counter:
        print("\nTop chosen actions:")
        for key, count in choice_counter.most_common(args.top_n):
            print(f"- {key}: {count}")

    changed_counter = Counter(
        f"{r.get('default_action')} -> {r.get('parsed_choice')}" for r in changed if r.get("parsed_choice")
    )
    if changed_counter:
        print("\nTop deviations from default:")
        for key, count in changed_counter.most_common(args.top_n):
            print(f"- {key}: {count}")

    print("\nSample changed decisions:")
    for row in changed[: args.top_n]:
        state = row.get("state", {})
        print(
            json.dumps(
                {
                    "sim_time": state.get("sim_time"),
                    "machine_type": state.get("machine_type"),
                    "candidates": row.get("candidates"),
                    "default_action": row.get("default_action"),
                    "parsed_choice": row.get("parsed_choice"),
                    "reason": row.get("reason"),
                    "usage": row.get("usage"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
