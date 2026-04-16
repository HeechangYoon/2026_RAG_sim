# JSSP RAG Assistant HF

This document describes the Hugging Face variant of the JSSP RAG assistant workflow.

## Purpose

Use the HF assistant path when you want local-model retrieval experiments instead of API-based interaction.

It is useful for:
- offline or local inspection of the DB
- prompt and retrieval debugging with a local model
- qualitative comparison between API and local-model answers

## Scope

The HF assistant is not the main benchmark runner.
Use it for retrieval-oriented QA and experimentation, not for the primary stepwise evaluation loop.

## Typical Inputs

- `jssp_rag.db`
- a local Hugging Face model identifier
- an optional prompt or payload file

## Typical Outputs

- local-model answer text
- optional retrieval context used to answer the query

## Notes

- Local HF models may behave differently from API models even with the same retrieved context.
- Use the HF assistant when you want to inspect retrieval behavior without depending on remote API calls.
- For main JSSP policy experiments, prefer `run_jssp_agent_step.py` and `run_jssp_threeway_eval.py`.
