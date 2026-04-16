# JSSP RAG Assistant

This document describes the lightweight question-answering helper built on top of `jssp_rag.db`.

## Purpose

`jssp_rag_assistant.py` is for interactive inspection of the JSSP RAG database.

It can be used to:
- inspect benchmark-derived knowledge stored in the DB
- ask natural-language questions about runs, bottlenecks, or decision cases
- test retrieval behavior outside the simulator loop

It is not the main stepwise evaluation runner.

## Typical Use Cases

- inspect what knowledge is stored in the DB
- ask which heuristic runs performed best on a given instance
- inspect retrieved logs or case summaries
- debug whether the DB contains the expected information before running agent experiments

## Input

The assistant typically reads from:
- `jssp_rag.db`
- optional JSON payloads such as files under `rag_payloads/`

## Output

The assistant returns a text answer based on retrieved DB content.
Depending on configuration, it may also show the retrieval context used to answer the question.

## Typical Workflow

1. Build the DB with `jssp_rag_db.py`
2. Prepare a query or JSON payload
3. Run `jssp_rag_assistant.py`
4. Inspect the answer and the retrieved context

## Notes

- This assistant is useful for DB inspection and retrieval debugging.
- It is separate from the stepwise control loop used in `run_jssp_agent_step.py`.
- If retrieval looks wrong here, it will likely also be wrong inside the stepwise agent.
