# JSSP RAG DB

This document describes how `jssp_rag_db.py` builds and refreshes the SQLite database used by the JSSP retrieval workflow.

## Purpose

`jssp_rag_db.py` is the only script that should create or modify `jssp_rag.db`.

It is responsible for:
- loading benchmark instances from `DT/data/raw/JSSP`
- converting raw benchmark files to unified CSV when needed
- running heuristic stepwise simulations
- storing event logs, decision traces, and retrieval-ready decision cases
- tagging teacher-quality runs and cases for retrieval

`run_jssp_agent_step.py` and `run_jssp_threeway_eval.py` should be treated as read-only consumers of the DB.

## Input Paths

- Raw benchmark instances: `DT/data/raw/JSSP`
- Preprocessed unified CSV files: `DT/data/preprocessed`
- Default DB file: `jssp_rag.db`
- Default output directory used during DB generation: `DT/results/JSSP_RAG`

## Main Tables

- `instances`: instance metadata
- `runs`: one row per heuristic run, including `teacher_tier`
- `job_metrics`: job completion metrics
- `operation_metrics`: operation timing metrics
- `machine_metrics`: machine utilization metrics
- `rag_documents`: run-level summary documents
- `run_logs`: full event log text
- `rag_log_chunks`: chunked event logs for fallback retrieval
- `rag_decision_memories`: compact run-level decision summaries
- `decision_traces`: stepwise heuristic decisions
- `decision_cases`: dispatch-focused retrieval cases built from decision traces
- `instance_features`: problem-level retrieval features

## Decision Cases

`decision_cases` are the main retrieval unit for dispatch decisions.

Each case stores:
- problem and run identity
- decision time and machine context
- candidate set and selected action
- compact state summary
- local candidate features
- global workload summary
- final outcome summary
- teacher tier

Current local retrieval features include:
- candidate count
- candidate profile vector `(proc_time_on_machine, remaining_work)`
- chosen candidate rank by proc time
- chosen candidate rank by remaining work
- bottleneck queue estimate
- global remaining-work summary

## How RAG Changes the Prompt

The DB does not generate a completely different prompt template.
Instead, retrieval appends compact case summaries to the base dispatch prompt used by `run_jssp_agent_step.py`.

Conceptually:

```text
base dispatch prompt
+ retrieved decision cases
= final prompt used for LLM+RAG
```

Without RAG, the model sees only the current simulator state.

With RAG, the model sees the current simulator state plus compact summaries of similar teacher cases from `decision_cases`.

Those appended case summaries typically include:
- similarity score
- source instance
- machine context
- chosen action
- default action
- chosen/default candidate profile
- chosen rank by proc time
- chosen rank by remaining work
- compact summary of other candidates
- compact global remaining-work summary
- final makespan

This means the role of the DB is not to replace the base prompt.
Its role is to inject retrieval evidence into the existing prompt.

## Teacher Corpus Strategy

The DB can store many heuristic runs, but retrieval should usually prefer teacher-quality cases.

Supported teacher modes:
- `dispatch_rule`: mark runs as teacher if their dispatching rule is in `--teacher-dispatching-rules`
- `best_run`: mark the best run(s) per instance as teacher according to makespan

Useful CLI options:
- `--teacher-only`
- `--teacher-dispatching-rules SPT`
- `--teacher-mode dispatch_rule|best_run`
- `--teacher-top-k N`

## Typical Commands

Build an SPT-only teacher corpus:

```bash
python jssp_rag_db.py --teacher-only --teacher-dispatching-rules SPT --overwrite
```

Store multiple heuristics but mark only SPT as teacher:

```bash
python jssp_rag_db.py ^
  --dispatching-rules FIFO,SPT,MWKR,LWKR ^
  --teacher-dispatching-rules SPT ^
  --overwrite
```

Store multiple heuristics and mark the best run per instance as teacher:

```bash
python jssp_rag_db.py ^
  --dispatching-rules FIFO,SPT,MWKR,LWKR ^
  --teacher-mode best_run ^
  --teacher-top-k 1 ^
  --overwrite
```

## Notes

- Rebuild the DB after changing decision-case features or teacher-tier logic.
- Retrieval quality depends heavily on the quality of `decision_cases`, not just the size of the DB.
- If you later add RL or search-based teacher policies, the same teacher-tier mechanism can be reused.
