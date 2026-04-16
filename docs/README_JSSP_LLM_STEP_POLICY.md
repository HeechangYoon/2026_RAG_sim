# JSSP LLM Step Policy

This document is a high-level note about the stepwise LLM policy workflow.

## Current Preferred Entry Point

For current experiments, use:
- `run_jssp_agent_step.py` for single-instance evaluation
- `run_jssp_threeway_eval.py` for batch comparison

These scripts are the maintained entry points.

## Current Policy Shape

The current dispatch-oriented LLM policy is designed to:
- use a strong heuristic baseline such as `SPT`
- keep the default action unless there is a clear reason to override it
- consider candidate processing time, remaining work, and bottleneck congestion
- use RAG decision cases as teacher-style references when enabled

## Retrieval Design

Current retrieval is based on `decision_cases` in `jssp_rag.db`.

The retrieval signal combines:
- problem-level similarity
- local candidate-structure similarity
- global remaining-work similarity
- teacher-tier filtering

## Practical Goal

The immediate goal is not unlimited override behavior.
The goal is controlled improvement over a strong heuristic baseline.

A healthy policy usually looks like this:
- low invalid-action rate
- low unnecessary override rate
- makespan close to or better than `SPT`
- stable behavior across multiple benchmark instances

## Recommended Workflow

1. Build or refresh the teacher DB with `jssp_rag_db.py`
2. Run `run_jssp_agent_step.py` on a small instance for debugging
3. Use `run_jssp_threeway_eval.py` for broader comparison
4. Analyze failures and override impact with the analysis scripts

## Notes on Teacher Quality

If retrieval uses only weak or noisy cases, the LLM will usually not improve over a good heuristic.
For that reason:
- `SPT` is a useful stability-oriented teacher
- best-run teacher selection can be useful when stronger teachers are available
- RL or search-generated teachers can be added later using the same DB structure
