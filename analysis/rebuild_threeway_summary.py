import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild JSSP three-way evaluation summary.csv from saved logs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("DT") / "results" / "JSSP_BATCH_EVAL",
        help="Directory containing *_spt.csv, *_llm.csv, *_llm_rag.csv and decision logs.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output summary CSV path. Defaults to <results-dir>/summary.csv.",
    )
    return parser.parse_args()


def _event_makespan(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    df.columns = [str(col).strip().lower() for col in df.columns]
    required = {"event", "time", "part_id"}
    if not required.issubset(df.columns):
        return None
    df["event"] = df["event"].astype(str).str.lower()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    completed = df[df["event"].isin(["job transferred to sink", "job completed"])].copy()
    if completed.empty:
        return None
    grouped = completed.groupby("part_id", as_index=False)["time"].max()
    if grouped.empty:
        return None
    try:
        return float(grouped["time"].max())
    except Exception:
        return None


def _last_jsonl_record(path: Path) -> dict:
    if not path.exists():
        return {}
    last_line = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line.strip()
    except Exception:
        return {}
    if not last_line:
        return {}
    try:
        parsed = json.loads(last_line)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _reconstruct_error(decision_log_path: Path, makespan: float | None) -> str:
    if makespan is not None:
        return ""
    last = _last_jsonl_record(decision_log_path)
    query_error = str(last.get("query_error") or "").strip()
    error = str(last.get("error") or "").strip()
    if query_error:
        return f"reconstructed_error: query_error='{query_error}'"
    if error:
        return f"reconstructed_error: error='{error}'"
    if decision_log_path.exists():
        return "reconstructed_error: decision log exists but final error is unavailable"
    return ""


def _instance_names(results_dir: Path) -> list[str]:
    names = set()
    for path in results_dir.iterdir():
        name = path.name
        for suffix in [
            "_spt.csv",
            "_llm.csv",
            "_llm_rag.csv",
            "_llm_decision.jsonl",
            "_llm_rag_decision.jsonl",
        ]:
            if name.endswith(suffix):
                names.add(name[: -len(suffix)])
                break
    return sorted(names)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")
    summary_csv = args.summary_csv or (results_dir / "summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for instance in _instance_names(results_dir):
        spt_log = results_dir / f"{instance}_spt.csv"
        llm_log = results_dir / f"{instance}_llm.csv"
        rag_log = results_dir / f"{instance}_llm_rag.csv"
        llm_decision = results_dir / f"{instance}_llm_decision.jsonl"
        rag_decision = results_dir / f"{instance}_llm_rag_decision.jsonl"

        spt_makespan = _event_makespan(spt_log)
        llm_makespan = _event_makespan(llm_log)
        rag_makespan = _event_makespan(rag_log)

        row = {
            "instance": instance,
            "spt_makespan": spt_makespan,
            "llm_makespan": llm_makespan,
            "llm_rag_makespan": rag_makespan,
            "delta_llm_vs_spt": (llm_makespan - spt_makespan) if llm_makespan is not None and spt_makespan is not None else None,
            "delta_llm_rag_vs_spt": (rag_makespan - spt_makespan) if rag_makespan is not None and spt_makespan is not None else None,
            "delta_llm_rag_vs_llm": (rag_makespan - llm_makespan) if rag_makespan is not None and llm_makespan is not None else None,
            "spt_error": "",
            "llm_error": _reconstruct_error(llm_decision, llm_makespan),
            "llm_rag_error": _reconstruct_error(rag_decision, rag_makespan),
            "spt_log": str(spt_log),
            "llm_log": str(llm_log),
            "llm_rag_log": str(rag_log),
            "llm_decision_log": str(llm_decision),
            "llm_rag_decision_log": str(rag_decision),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")
    print(f"rows={len(df)}")


if __name__ == "__main__":
    main()
