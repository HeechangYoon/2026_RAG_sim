import argparse
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


# Extract probable benchmark names from user question (e.g., la01, ft06, ta21).
INSTANCE_PATTERN = re.compile(r"([a-z]{2,}\d{1,3})", re.IGNORECASE)
SYSTEM_PROMPT = (
    "You are a scheduling analyst. "
    "Answer only using provided context. "
    "If evidence is insufficient, state uncertainty clearly."
)


@dataclass
class RAGConfig:
    db_path: Path
    llm_api_url: str
    llm_api_key: str
    llm_model: str
    temperature: float
    max_tokens: int
    top_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG assistant over jssp_rag.db")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--db-path", type=Path, default=Path("jssp_rag.db"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--llm-api-url", default=os.getenv("LLM_API_URL", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--dry-run", action="store_true", help="Print prompt context only, skip API call")
    parser.add_argument(
        "--show-augmented-query",
        action="store_true",
        help="Print the final RAG-augmented payload (messages) sent to LLM API",
    )
    parser.add_argument(
        "--save-augmented-path",
        type=Path,
        default=None,
        help="Optional file path to save the final RAG-augmented payload JSON",
    )
    return parser.parse_args()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1", (table_name,)
    ).fetchone()
    return row is not None


def extract_instance_candidates(question: str) -> list[str]:
    return sorted({m.group(1).lower() for m in INSTANCE_PATTERN.finditer(question)})


def retrieve_documents(
    conn: sqlite3.Connection,
    question: str,
    top_k: int,
    instance_names: list[str] | None = None,
) -> list[sqlite3.Row]:
    instance_names = instance_names or []

    # 1) Preferred path: full-text retrieval from FTS index when available.
    if table_exists(conn, "rag_documents_fts"):
        base_query = """
            SELECT rd.doc_id, rd.instance_name, rd.title, rd.content, rd.metadata_json
            FROM rag_documents_fts f
            JOIN rag_documents rd ON rd.doc_id = f.rowid
            WHERE rag_documents_fts MATCH ?
        """
        params: list[object] = [question]
        if instance_names:
            placeholders = ",".join("?" for _ in instance_names)
            base_query += f" AND lower(rd.instance_name) IN ({placeholders})"
            params.extend(n.lower() for n in instance_names)
        base_query += " LIMIT ?"
        params.append(top_k)
        try:
            rows = conn.execute(base_query, params).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            # If FTS query syntax fails (or extension mismatch), fallback below.
            pass

    # 2) Fallback path: LIKE retrieval on raw table.
    like_q = f"%{question}%"
    query = """
        SELECT doc_id, instance_name, title, content, metadata_json
        FROM rag_documents
        WHERE title LIKE ? OR content LIKE ?
    """
    params2: list[object] = [like_q, like_q]
    if instance_names:
        placeholders = ",".join("?" for _ in instance_names)
        query += f" AND lower(instance_name) IN ({placeholders})"
        params2.extend(n.lower() for n in instance_names)
    query += " ORDER BY doc_id DESC LIMIT ?"
    params2.append(top_k)
    rows = conn.execute(query, params2).fetchall()
    if rows:
        return rows

    # 3) Last resort: recent docs, optionally filtered by instance.
    fallback_query = """
        SELECT doc_id, instance_name, title, content, metadata_json
        FROM rag_documents
    """
    params3: list[object] = []
    if instance_names:
        placeholders = ",".join("?" for _ in instance_names)
        fallback_query += f" WHERE lower(instance_name) IN ({placeholders})"
        params3.extend(n.lower() for n in instance_names)
    fallback_query += " ORDER BY doc_id DESC LIMIT ?"
    params3.append(top_k)
    return conn.execute(fallback_query, params3).fetchall()


def retrieve_best_runs(conn: sqlite3.Connection, instance_names: list[str], limit_per_instance: int = 5) -> list[sqlite3.Row]:
    # Provide compact "answer-first" numeric evidence (best makespan candidates).
    if instance_names:
        placeholders = ",".join("?" for _ in instance_names)
        query = f"""
            SELECT i.name AS instance_name, r.sequencing_rule, r.routing_rule, r.dispatching_rule, r.makespan
            FROM runs r
            JOIN instances i ON i.instance_id = r.instance_id
            WHERE r.status = 'success'
              AND lower(i.name) IN ({placeholders})
            ORDER BY i.name, r.makespan ASC
            LIMIT ?
        """
        params = [n.lower() for n in instance_names] + [max(1, limit_per_instance * len(instance_names))]
        return conn.execute(query, params).fetchall()

    return conn.execute(
        """
        SELECT i.name AS instance_name, r.sequencing_rule, r.routing_rule, r.dispatching_rule, r.makespan
        FROM runs r
        JOIN instances i ON i.instance_id = r.instance_id
        WHERE r.status = 'success'
        ORDER BY r.makespan ASC
        LIMIT ?
        """,
        (max(10, limit_per_instance),),
    ).fetchall()


def retrieve_similar_log_chunks(
    conn: sqlite3.Connection,
    question: str,
    top_k: int,
    instance_names: list[str] | None = None,
) -> list[sqlite3.Row]:
    instance_names = instance_names or []
    if not table_exists(conn, "rag_log_chunks"):
        return []

    if table_exists(conn, "rag_log_chunks_fts"):
        base_query = """
            SELECT c.chunk_id, c.run_id, c.instance_name, c.chunk_index, c.content, c.metadata_json
            FROM rag_log_chunks_fts f
            JOIN rag_log_chunks c ON c.chunk_id = f.rowid
            WHERE rag_log_chunks_fts MATCH ?
        """
        params: list[object] = [question]
        if instance_names:
            placeholders = ",".join("?" for _ in instance_names)
            base_query += f" AND lower(c.instance_name) IN ({placeholders})"
            params.extend(n.lower() for n in instance_names)
        base_query += " LIMIT ?"
        params.append(top_k)
        try:
            rows = conn.execute(base_query, params).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass

    like_q = f"%{question}%"
    query = """
        SELECT chunk_id, run_id, instance_name, chunk_index, content, metadata_json
        FROM rag_log_chunks
        WHERE content LIKE ?
    """
    params2: list[object] = [like_q]
    if instance_names:
        placeholders = ",".join("?" for _ in instance_names)
        query += f" AND lower(instance_name) IN ({placeholders})"
        params2.extend(n.lower() for n in instance_names)
    query += " ORDER BY chunk_id DESC LIMIT ?"
    params2.append(top_k)
    rows = conn.execute(query, params2).fetchall()
    if rows:
        return rows

    fallback_query = """
        SELECT chunk_id, run_id, instance_name, chunk_index, content, metadata_json
        FROM rag_log_chunks
    """
    params3: list[object] = []
    if instance_names:
        placeholders = ",".join("?" for _ in instance_names)
        fallback_query += f" WHERE lower(instance_name) IN ({placeholders})"
        params3.extend(n.lower() for n in instance_names)
    fallback_query += " ORDER BY chunk_id DESC LIMIT ?"
    params3.append(top_k)
    return conn.execute(fallback_query, params3).fetchall()


def build_context(
    question: str,
    docs: list[sqlite3.Row],
    log_chunks: list[sqlite3.Row],
    best_runs: list[sqlite3.Row],
) -> str:
    # This context is the actual RAG-augmented user message sent to LLM.
    doc_lines = []
    for idx, row in enumerate(docs, start=1):
        doc_lines.append(
            f"[DOC {idx}] instance={row['instance_name']}\n"
            f"title={row['title']}\n"
            f"content={row['content']}\n"
        )

    run_lines = []
    for idx, row in enumerate(best_runs, start=1):
        run_lines.append(
            f"[RUN {idx}] instance={row['instance_name']}, "
            f"seq={row['sequencing_rule']}, route={row['routing_rule']}, "
            f"dispatch={row['dispatching_rule']}, makespan={row['makespan']}"
        )

    chunk_lines = []
    for idx, row in enumerate(log_chunks, start=1):
        chunk_lines.append(
            f"[CHUNK {idx}] instance={row['instance_name']}, "
            f"run_id={row['run_id']}, chunk_index={row['chunk_index']}\n"
            f"{row['content']}\n"
        )

    return (
        "### USER QUESTION\n"
        f"{question}\n\n"
        "### SIMILAR LOG CHUNKS\n"
        + ("\n".join(chunk_lines) if chunk_lines else "No matched log chunks.\n")
        + "\n"
        "### RETRIEVED DOCUMENTS\n"
        + ("\n".join(doc_lines) if doc_lines else "No matched documents.\n")
        + "\n### BEST RUNS SNAPSHOT\n"
        + ("\n".join(run_lines) if run_lines else "No run rows.\n")
    )


def call_openai_compatible_chat(
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    user_prompt: str,
) -> str:
    # OpenAI-compatible Chat Completions payload.
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM API HTTPError {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API connection error: {exc}") from exc

    parsed = json.loads(body)
    choices = parsed.get("choices", [])
    if not choices:
        raise RuntimeError(f"Unexpected API response: {body}")
    return choices[0]["message"]["content"]


def build_augmented_payload(model: str, temperature: float, max_tokens: int, context: str) -> dict:
    # Exposed separately so caller can inspect/save payload before API call.
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
    }


def main() -> None:
    args = parse_args()
    config = RAGConfig(
        db_path=args.db_path,
        llm_api_url=args.llm_api_url.strip(),
        llm_api_key=args.llm_api_key.strip(),
        llm_model=args.llm_model.strip(),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
    )

    if not config.db_path.exists():
        raise SystemExit(f"DB file not found: {config.db_path}")

    conn = sqlite3.connect(str(config.db_path))
    conn.row_factory = sqlite3.Row

    # Retrieval pipeline: instance hint extraction -> doc search -> best-run snapshot.
    instance_candidates = extract_instance_candidates(args.question)
    log_chunks = retrieve_similar_log_chunks(conn, args.question, config.top_k, instance_candidates)
    docs = retrieve_documents(conn, args.question, config.top_k, instance_candidates)
    best_runs = retrieve_best_runs(conn, instance_candidates)
    context = build_context(args.question, docs, log_chunks, best_runs)
    augmented_payload = build_augmented_payload(
        model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        context=context,
    )

    if args.show_augmented_query:
        print("=== AUGMENTED QUERY (LLM PAYLOAD) ===")
        print(json.dumps(augmented_payload, ensure_ascii=False, indent=2))

    if args.save_augmented_path is not None:
        # Persisting this is useful for debugging retrieval quality/regressions.
        args.save_augmented_path.parent.mkdir(parents=True, exist_ok=True)
        args.save_augmented_path.write_text(
            json.dumps(augmented_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved augmented payload: {args.save_augmented_path}")

    if args.dry_run:
        print(context)
        conn.close()
        return

    if not config.llm_api_url:
        conn.close()
        raise SystemExit("Set --llm-api-url or LLM_API_URL.")
    if not config.llm_api_key:
        conn.close()
        raise SystemExit("Set --llm-api-key or LLM_API_KEY.")

    answer = call_openai_compatible_chat(
        api_url=config.llm_api_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        user_prompt=context,
    )
    conn.close()

    print("=== ANSWER ===")
    print(answer)
    print("\n=== RETRIEVED INSTANCES ===")
    for row in docs:
        print(f"- {row['instance_name']} | {row['title']}")


if __name__ == "__main__":
    main()
