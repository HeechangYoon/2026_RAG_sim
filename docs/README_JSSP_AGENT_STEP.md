# JSSP Agent Step Runner

This document describes `run_jssp_agent_step.py`, the stepwise simulation runner used for single-instance experiments.

## Purpose

`run_jssp_agent_step.py` runs one JSSP instance with one selected agent configuration.

It supports:
- heuristic decisions
- OpenAI-compatible API LLM decisions
- local Hugging Face model decisions
- placeholder RL-style decisions

It should read `jssp_rag.db`. It should not build or update the DB.

## Supported Agent Types

- `heuristic`
- `llm`
- `hf`
- `rl`

## Supported Decision Types

- `sequencing`
- `routing`
- `dispatch`

Use `--agent-decision-types` to restrict which decision types are handled by the agent.

Examples:
- `--agent-decision-types dispatch`
- `--agent-decision-types routing,dispatch`
- `--agent-decision-types all`

## Basic Example

Run a dispatch-only LLM experiment with RAG:

```bash
python run_jssp_agent_step.py ^
  --agent-type llm ^
  --instance la01 ^
  --sequencing-rule FIFO ^
  --routing-rule FIFO ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch ^
  --use-rag ^
  --rag-db-path jssp_rag.db ^
  --event-log-path DT/results/JSSP_AGENT/la01_llm_dispatch.csv ^
  --decision-log-path DT/results/JSSP_AGENT/decision_trace_dispatch_llm.jsonl ^
  --prompt-log-path DT/results/JSSP_AGENT/prompt_trace_dispatch_llm.jsonl ^
  --show-prompts
```

Run the same experiment without RAG:

```bash
python run_jssp_agent_step.py ^
  --agent-type llm ^
  --instance la01 ^
  --sequencing-rule FIFO ^
  --routing-rule FIFO ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch ^
  --no-use-rag ^
  --event-log-path DT/results/JSSP_AGENT/la01_llm_dispatch_no_rag.csv ^
  --decision-log-path DT/results/JSSP_AGENT/decision_trace_dispatch_llm_no_rag.jsonl
```

Run the heuristic baseline:

```bash
python run_jssp_agent_step.py ^
  --agent-type heuristic ^
  --instance la01 ^
  --sequencing-rule FIFO ^
  --routing-rule FIFO ^
  --dispatching-rule SPT ^
  --event-log-path DT/results/JSSP_AGENT/la01_spt.csv
```

## Important LLM Options

- `--api-url`
- `--api-key`
- `--model`
- `--temperature`
- `--max-tokens`
- `--timeout-sec`
- `--adaptive-max-tokens`
- `--adaptive-token-step`
- `--adaptive-token-cap`
- `--use-rag` / `--no-use-rag`
- `--rag-db-path`
- `--rag-top-k`

## Adaptive Token Budget

If prompt size increases with problem size, you can scale the output budget automatically.

Example:

```bash
python run_jssp_agent_step.py ^
  --agent-type llm ^
  --instance la01 ^
  --max-tokens 800 ^
  --adaptive-max-tokens ^
  --adaptive-token-step 200 ^
  --adaptive-token-cap 1600
```

This keeps a fixed base budget for small instances and increases the cap for larger ones.

## Logging

Useful output files:
- event log CSV
- decision trace JSONL
- prompt trace JSONL
- optional Gantt chart image

Helpful options:
- `--decision-log-path`
- `--prompt-log-path`
- `--reset-log-files`
- `--show-prompts`
- `--save-gantt`

## Prompt Structure

The step runner uses a base dispatch prompt and optionally appends retrieved decision cases.

### Base Dispatch Prompt Without RAG

At a high level, the dispatch prompt contains:
- objective
- compact problem summary
- compact global workload summary
- recent decisions
- current machine and simulation time
- candidate job ids
- default action
- candidate statistics
- bottleneck summary
- strict JSON output instruction

Representative shape:

```text
Objective: minimize final makespan for this dispatch decision.
Problem summary: {"total_jobs": 10, "completed_jobs": 0, "remaining_jobs": 10}
Global remaining-work summary: {"top_rem_work":[413.0,370.0,354.0,330.0,246.0],"avg_rem_work_top":322.6,"near_finish_jobs":0}
Recent decisions: t=39.0 M4 d=J9 c=J9 | t=56.0 M4 d=J2 c=J2 | t=74.0 M1 d=J8 c=J8
Machine: M4 at time 108.0
Candidates: J7, J10
Default action: J7
Candidate stats: J7: p=69.0, rem_work=413.0, rem_ops=5 | J10: p=79.0, rem_work=293.0, rem_ops=4
Bottleneck summary: [...]
Choose one job id.
Favor lower proc_time_on_machine and lower remaining_work on bottleneck machines.
Use remaining_ops only as a tie-breaker, not the main criterion.
Do not pick a job only because it has fewer remaining_ops.
Return only JSON. Example: {"choice": "J7", "reason": "short explanation"}
```

### Dispatch Prompt With RAG

When `--use-rag` is enabled, the same base prompt is used and one extra section is appended:

```text
Retrieved best prior cases:
[DECISION_CASE] ...
[DECISION_CASE] ...
```

The current implementation does not replace the original prompt. It augments it.

### What Retrieval Adds

Each retrieved case is a compact teacher-style summary of a prior dispatch decision.

A retrieved case includes:
- similarity score
- source instance
- machine context
- number of candidates
- chosen action and default action
- chosen candidate profile
- default candidate profile
- chosen candidate ranks within the candidate set
- compact summary of the other candidate range
- compact global remaining-work summary
- bottleneck queue
- final makespan

Representative shape:

```text
[DECISION_CASE] sim_score=8.296 instance=ta31
- machine=M8 candidates=2
- chosen_action=J7 default_action=J7
- chosen_profile={'job_id': 'J7', 'proc_time': 70.0, 'remaining_work': 249.0, 'remaining_ops': 4}
- default_profile={'job_id': 'J7', 'proc_time': 70.0, 'remaining_work': 249.0, 'remaining_ops': 4}
- chosen_ranks: proc=0 rem_work=1
- other_candidates={'proc': [71.0, 71.0], 'rem_work': [196.0, 196.0], 'count': 1}
- global_top3_rem_work=[318.0, 311.0, 298.0]
- bottleneck_queue=0.0 makespan=2130.0
```

### Why This Matters

Without RAG, the model only sees the current state.

With RAG, the model sees:
- the current state
- plus compact teacher cases that show what was chosen in similar situations and how that case ended

This is the main prompt-level difference between `LLM` and `LLM+RAG`.

## Three-Way Batch Evaluation

Use `run_jssp_threeway_eval.py` for batch comparison of:
- `SPT`
- `LLM`
- `LLM+RAG`

Example with explicit instances:

```bash
python run_jssp_threeway_eval.py ^
  --instances la01,la02 ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch ^
  --rag-db-path jssp_rag.db
```

If `--instances` is omitted, instances are auto-discovered from `DT/data/raw/JSSP`.

```bash
python run_jssp_threeway_eval.py ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch ^
  --rag-db-path jssp_rag.db
```

Limit the run to the first few auto-discovered instances:

```bash
python run_jssp_threeway_eval.py ^
  --limit 5 ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch ^
  --rag-db-path jssp_rag.db
```

Continue even if one LLM call fails:

```bash
python run_jssp_threeway_eval.py ^
  --continue-on-error ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch ^
  --rag-db-path jssp_rag.db
```

Stop immediately on the first failure:

```bash
python run_jssp_threeway_eval.py ^
  --no-continue-on-error ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch ^
  --rag-db-path jssp_rag.db
```

Use adaptive token scaling in batch evaluation:

```bash
python run_jssp_threeway_eval.py ^
  --adaptive-max-tokens ^
  --adaptive-token-step 200 ^
  --adaptive-token-cap 1600 ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch ^
  --rag-db-path jssp_rag.db
```

## Four-Way Batch Evaluation

Use `run_jssp_fourway_eval.py` when you want to compare:
- `SPT`
- `LLM`
- `LLM+RAG(benchmark DB)`
- `LLM+RAG(synthetic DB)`

This is useful when benchmark-derived retrieval is treated as a reference or upper bound, while synthetic retrieval is the main hypothesis test.

Example:

```bash
python run_jssp_fourway_eval.py ^
  --benchmark-rag-db-path jssp_rag_benchmark.db ^
  --synthetic-rag-db-path jssp_rag_synthetic.db ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch
```

Repeat each method multiple times and aggregate by instance:

```bash
python run_jssp_fourway_eval.py ^
  --benchmark-rag-db-path jssp_rag_benchmark.db ^
  --synthetic-rag-db-path jssp_rag_synthetic.db ^
  --num-repeats 5 ^
  --runs-csv DT/results/JSSP_FOURWAY_EVAL/runs.csv ^
  --summary-csv DT/results/JSSP_FOURWAY_EVAL/summary.csv ^
  --dispatching-rule SPT ^
  --agent-decision-types dispatch
```

In repeat mode:
- `runs.csv` stores one row per `(instance, repeat)`
- `summary.csv` stores aggregated statistics per instance

Representative aggregated fields:
- `*_mean`
- `*_std`
- `*_min`
- `*_max`
- `*_count`
- `bench_better_than_llm_rate`
- `synth_better_than_llm_rate`
- `synth_better_than_bench_rate`

Key summary columns:
- `spt_makespan_mean`
- `llm_makespan_mean`
- `llm_rag_bench_makespan_mean`
- `llm_rag_synth_makespan_mean`
- `delta_llm_vs_spt_mean`
- `delta_bench_vs_spt_mean`
- `delta_synth_vs_spt_mean`
- `delta_bench_vs_llm_mean`
- `delta_synth_vs_llm_mean`
- `delta_synth_vs_bench_mean`

## Related Analysis Scripts

- `analyze_decision_trace.py`
- `compare_event_logs.py`
- `analyze_override_impact.py`

Use them to compare decisions, makespan deltas, and the effect of LLM overrides.
