"""
rag_server.py — The real server. Wraps rag3.py's contract search and
ask_db.py's structured data queries so teammates' laptops can ask questions
over the network instead of touching the ChromaDB/SQLite files directly.

Endpoints:
    GET  /health         — is the server alive, how much data is loaded
    POST /ask/contracts   — semantic search over the carrier contracts
    POST /ask/data        — Text-to-SQL over a structured table
    POST /ask             — auto-routes to whichever of the above fits the question
    GET  /                — the chat UI teammates actually use (chat.html)

Run with:
    uvicorn rag_server:app --host 0.0.0.0 --port 8000

Then open http://<this laptop's IP>:8000 in a browser for the chat UI, or
use PowerShell's Invoke-RestMethod to hit the JSON endpoints directly.
"""

import os

# See rag3.py for why this is opt-in only, not on by default: disabling TLS
# certificate verification globally is a real security risk, not a style
# choice. Only set this if you're behind a corporate proxy that re-signs
# certificates and you've confirmed you actually need it.
if os.environ.get("ALLOW_INSECURE_SSL") == "1":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

import time
import shutil
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, File, UploadFile, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag3     # contract semantic search — nothing duplicated, just imported
import ask_db   # structured SQL data engine — same deal
import ask      # the routing heuristic (route_question / guess_table)

# CUSTOMIZE: rename this to your own project/team name.
app = FastAPI(title="Local Knowledge Assistant Server")

SERVER_START_TIME = time.time()

# ============================================================================
# >>> CUSTOMIZE HERE: register each of your SQL tables with a human-friendly
# description. This is used to prefix data-lookup answers (e.g. "According
# to the West Region rate table...") the same way contract answers already
# cite "[Source: filename.pdf, Page X]". Add one entry per table you connect
# in ask_db.py — the key must match the real table name in your database.
# ============================================================================
TABLE_SOURCE_LABELS = {
    "vendor_rates": "your vendor rate table",      # <-- rename/replace with your real table
    "shipments":    "your shipment/order export",  # <-- rename/replace with your real table
}


# ── Access control (task #14) ────────────────────────────────────────────────
# Nothing below actually WRITES to the knowledge base yet — today's endpoints
# are all read-only questions, so there's nothing to lock down yet. But when
# we add endpoints that add/replace/delete a contract (the "quarterly refresh"
# workflow), those should require this key. Attach `admin=Depends(require_admin_key)`
# as a parameter to any future write endpoint to protect it.
#
# The key is read from an environment variable, NOT hardcoded here, so it's
# not sitting in a file that gets shared/viewed by the whole team. Set it
# once per admin laptop with (PowerShell):
#   $env:RAG_ADMIN_KEY = "choose-a-real-secret-here"
# before starting the server. Everyone querying (read-only) needs no key at all.
ADMIN_KEY = os.environ.get("RAG_ADMIN_KEY")


def require_admin_key(x_admin_key: str = Header(default=None)):
    if not ADMIN_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server has no RAG_ADMIN_KEY configured — admin actions are disabled until one is set."
        )
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing admin key.")
    return True


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    history: list[dict] | None = None  # [{"question": ..., "answer": ...}, ...], oldest first


class DataQuestionRequest(BaseModel):
    table: str
    question: str
    history: list[dict] | None = None


class AutoQuestionRequest(BaseModel):
    question: str
    table: str | None = None  # optional override, same as ask.py's --table
    history: list[dict] | None = None


# ── Ask-for-clarification instead of guessing (vague carrier questions) ─────
# A question like "what is the termination notice period?" with no carrier
# named is genuinely ambiguous once the knowledge base has more than one
# carrier loaded — every carrier has SOME answer to that, and they're not
# all identical (see the Vendor B/Vendor C governing-law outliers). Rather
# than retrieving across everything and letting the LLM either blend
# carriers together or arbitrarily favor whichever chunk scored highest,
# detect this case in code (same deterministic-detection pattern as the
# meta-question and carrier-filter logic above) and ask the user which
# carrier they mean — a real clarifying question, not a guess.
CLAUSE_TOPIC_KEYWORDS = [
    "termination", "governing law", "liability", "insurance", "claim",
    "payment terms", "rate", "notice period", "indemnif", "cargo",
    "insurance minimum",
]


def carrier_display_name(entry: dict) -> str:
    """Turn a document's stored name into a short, human label for a
    clarification chip — e.g. 'Vendor D' instead of the raw filename
    '(Executed) Vendor_D_Master_Agreement_2026.pdf'.
    Reuses the same stopword-filtering token extractor already built for
    carrier-name filtering, so the two stay consistent with each other."""
    tokens = rag3.extract_carrier_tokens(entry.get("name", ""))
    return " ".join(tokens) if tokens else entry.get("name", "Unknown")


def _needs_carrier_clarification(question: str, index: dict) -> bool:
    q_lower = question.lower()
    docs = index.get("documents", {})
    if len(docs) <= 1:
        return False  # nothing to disambiguate — only one carrier loaded
    mentions_clause_topic = any(ask._contains_keyword(q_lower, kw) for kw in CLAUSE_TOPIC_KEYWORDS)
    if not mentions_clause_topic:
        return False
    return rag3.detect_carrier_filter(question, index) is None


def _clarification_response(index: dict, resolved_question: str) -> dict:
    options = sorted({carrier_display_name(entry) for entry in index.get("documents", {}).values()})
    lines = "\n".join(f"- {name}" for name in options)
    answer = (
        "That could apply to any of the carriers in the knowledge base, and their terms "
        "aren't all the same — which carrier's contract are you asking about?\n\n"
        f"{lines}"
    )
    # resolved_question is the history-folded version, not the bare current
    # message — e.g. for a pronoun follow-up like "what are those?" this is
    # "<previous question> what are those?", not just "what are those?" on
    # its own. Returning it lets the frontend build a coherent chip click
    # ("<previous question> what are those? for Vendor B") instead of losing the
    # real context and sending a nonsense standalone question — see the bug
    # this fixes in _fold_in_history's docstring below.
    return {"needs_clarification": True, "answer": answer, "options": options,
            "resolved_question": resolved_question}


FOLLOWUP_START_PHRASES = ["what about", "how about", "and what", "and how", "what if"]
FOLLOWUP_REFERENCE_WORDS = ["it", "that", "those", "this", "them", "same", "again", "also", "too", "either"]
FOLLOWUP_MAX_WORDS = 8  # a long, fully-formed question is unlikely to depend on the prior turn


def _is_dependent_followup(question: str) -> bool:
    """
    Decide whether this question actually NEEDS the prior turn's context, or
    stands on its own. Without this check, _fold_in_history used to glue the
    previous question onto EVERY new message unconditionally — fine for a
    real follow-up ("what about Vendor D?"), but actively harmful for an
    unrelated next question, since it would drag a stale, irrelevant topic
    into retrieval and routing for a question that has nothing to do with
    it. Only fold when there's an actual signal of dependency: the message
    starts with a follow-up phrase ("what about...", "and what...") or is
    short AND leans on a reference word ("it", "those", "that") instead of
    naming its own subject.
    """
    q_lower = question.lower().strip()
    if any(q_lower.startswith(p) for p in FOLLOWUP_START_PHRASES):
        return True
    words = q_lower.split()
    is_short = len(words) <= FOLLOWUP_MAX_WORDS
    has_reference_word = any(w.strip("?.,!") in FOLLOWUP_REFERENCE_WORDS for w in words)
    return is_short and has_reference_word


def _fold_in_history(question: str, history: list | None) -> str:
    """
    For a genuine follow-up like "what about <a specific document name>?" or "what are
    those?", the bare question alone often isn't enough — it has no idea
    what "those" or "it" refers to. Folding in the most recent prior
    question gives both retrieval AND the data-vs-contracts ROUTING
    DECISION real context to work with.

    But only do this when the question actually looks dependent
    (_is_dependent_followup) — a fresh, unrelated question should be
    treated as fresh, not have a stale prior topic glued onto it. E.g. after
    asking about Vendor A's termination clause, asking "what are the rates for
    New Jersey?" is a complete, independent question and should search
    accordingly, not carry Vendor A's termination context along with it.
    """
    if not history or not _is_dependent_followup(question):
        return question
    last_question = (history[-1].get("question") or "").strip()
    return f"{last_question} {question}" if last_question else question


# ── Core logic, reused by both the dedicated and the auto-routed endpoints ──
def _answer_contract_question(question: str, top_k: int = 5, history: list | None = None) -> dict:
    collection = rag3.get_collection()
    if collection.count() == 0:
        return {"error": "Knowledge base is empty."}

    retrieval_text = _fold_in_history(question, history)

    # Load the contract index once — used for the clarification check, the
    # carrier-name filtering below, and the key-terms grounding block further
    # down.
    index = rag3.load_index()

    if _needs_carrier_clarification(retrieval_text, index):
        return _clarification_response(index, retrieval_text)

    embedder = rag3.get_embedder()
    q_vector = embedder.encode([retrieval_text]).tolist()

    # If the question clearly names one carrier (task #5), restrict the
    # ChromaDB query to ONLY that document's chunks. This eliminates any
    # chance of a different carrier's terms getting retrieved alongside the
    # one actually asked about — increasingly important now that the
    # knowledge base is growing past a handful of documents with similar
    # boilerplate clauses.
    carrier_filter = rag3.detect_carrier_filter(retrieval_text, index)

    query_kwargs = dict(
        query_embeddings=q_vector,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    if carrier_filter:
        query_kwargs["where"] = {"source": {"$eq": carrier_filter}}

    results = collection.query(**query_kwargs)

    docs_found = results["documents"][0]
    metas      = results["metadatas"][0]
    distances  = results["distances"][0]

    if not docs_found:
        return {"answer": "No relevant chunks found.", "sources": []}

    context_parts = []
    sources        = []
    seen_pages     = set()

    for doc, meta, dist in zip(docs_found, metas, distances):
        score  = round(1 - dist, 3)
        page   = meta.get("page")
        source = meta.get("source")
        name   = meta.get("name", "?")
        sources.append({"name": name, "page": page, "similarity": score})

        page_key = (source, page)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)

        page_label = f", Page {page}" if page else ""
        if page is not None:
            page_chunks = collection.get(
                where={"$and": [{"source": {"$eq": source}}, {"page": {"$eq": page}}]},
                include=["documents", "metadatas"]
            )
            ordered = sorted(
                zip(page_chunks["documents"], page_chunks["metadatas"]),
                key=lambda pair: pair[1].get("chunk_id", 0)
            )
            full_page_text = "\n".join(d for d, _ in ordered)
        else:
            full_page_text = doc

        context_parts.append(f"[Source: {name}{page_label}]\n{full_page_text}")

    # Also surface each source document's auto-extracted "key terms" table
    # (governing law, liability cap/direction, claim deadlines, etc.) as
    # extra grounding, ahead of the raw retrieved paragraphs. This gives the
    # small local model a specific, structured anchor for high-stakes facts
    # instead of relying purely on it re-deriving them from prose each time.
    # Marked unverified since a human hasn't checked these values yet.
    # (Reuses the `index` already loaded above for the carrier filter.)
    seen_kt_srcs  = set()
    key_terms_parts = []
    for meta in metas:
        source = meta.get("source")
        if source in seen_kt_srcs:
            continue
        seen_kt_srcs.add(source)
        entry = index.get("documents", {}).get(source)
        if entry and entry.get("key_terms"):
            verified_note = "VERIFIED" if entry.get("key_terms_verified") else "NOT YET HUMAN-VERIFIED — use with appropriate caution"
            key_terms_parts.append(
                f"[Reference key terms for {entry.get('name', source)} — auto-extracted, {verified_note}]\n{entry['key_terms']}"
            )

    context = "\n\n---\n\n".join(context_parts)
    if key_terms_parts:
        context = "\n\n---\n\n".join(key_terms_parts) + "\n\n===\n\n" + context

    answer = rag3.query_ollama(question, context, history=history)

    filtered_carrier_name = None
    if carrier_filter:
        filtered_carrier_name = index.get("documents", {}).get(carrier_filter, {}).get("name")

    return {"answer": answer, "sources": sources, "carrier_filter_applied": filtered_carrier_name}


def _answer_data_question(table: str, question: str, history: list | None = None) -> dict:
    import pandas as pd

    conn    = ask_db.get_connection()
    columns = ask_db.get_columns(conn, table)
    if not columns:
        conn.close()
        return {"error": f"Table '{table}' not found."}

    relevant_columns = columns
    shortlisted      = False
    if len(columns) > ask_db.WIDE_TABLE_THRESHOLD:
        relevant_columns = ask_db.shortlist_relevant_columns(question, columns)
        shortlisted = True

    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    all_col_info = cur.fetchall()
    schema_text  = "\n".join(f"  {row[1]} ({row[2]})" for row in all_col_info
                              if row[1] in relevant_columns)

    col_list_sql   = ", ".join(relevant_columns)
    fullness_score = " + ".join(f"(CASE WHEN {c} IS NOT NULL THEN 1 ELSE 0 END)" for c in relevant_columns)
    cur.execute(
        f"SELECT {col_list_sql} FROM {table} "
        f"ORDER BY ({fullness_score}) DESC LIMIT 3"
    )
    sample_rows = cur.fetchall()
    sample_text = "\n".join(str(dict(zip(relevant_columns, row))) for row in sample_rows)

    MAX_ATTEMPTS     = 2
    previous_attempt = None
    sql_attempts     = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        sql = ask_db.query_ollama_for_sql(question, table, schema_text, sample_text, previous_attempt, history)
        sql_attempts.append(sql)

        if not sql.strip().upper().startswith("SELECT"):
            conn.close()
            return {
                "error": "Blocked: the generated query was not a SELECT statement.",
                "sql_attempts": sql_attempts,
            }

        try:
            df = pd.read_sql_query(sql, conn)
        except Exception as e:
            if attempt < MAX_ATTEMPTS:
                previous_attempt = {"sql": sql, "issue": f"This query raised an error: {e}"}
                continue
            conn.close()
            return {"error": f"SQL error: {e}", "sql_attempts": sql_attempts}

        if len(df) == 0:
            looks_suspicious = True
        elif len(df) == 1:
            row = df.iloc[0]
            looks_suspicious = row.isna().all() or (row.fillna(0) == 0).all()
        else:
            looks_suspicious = False

        if looks_suspicious and attempt < MAX_ATTEMPTS:
            previous_attempt = {
                "sql": sql,
                "issue": "This query returned 0 rows / all NULL, which is suspicious for this question. "
                         "Double-check the exact format of values shown in the SAMPLE ROWS."
            }
            continue

        conn.close()
        rows = df.to_dict(orient="records")
        # Turn the raw rows into a short natural-language sentence for the
        # chat UI, instead of it having to fall back to a raw JSON dump.
        # Clean the column labels first (strips SQL syntax like UPPER(...)
        # so the summarizer can't misread a function name as if it were
        # part of the data itself — see clean_column_label's docstring for
        # the real bug this fixes). The raw `rows` (with real SQL labels)
        # still get returned separately below for the "how this was
        # answered" debug view — only the summary uses the cleaned version.
        answer = ask_db.summarize_result_in_words(question, ask_db.clean_row_labels(rows))
        if answer:
            source_label = TABLE_SOURCE_LABELS.get(table, f"the '{table}' table")
            answer = f"According to {source_label}:\n\n{answer}"
        return {
            "answer": answer,
            "sql": sql,
            "sql_attempts": sql_attempts,
            "shortlisted_columns": relevant_columns if shortlisted else None,
            "row_count": len(df),
            "rows": rows,
        }

    conn.close()
    return {"error": "Could not get a confident result after retries.", "sql_attempts": sql_attempts}


# ── Meta-questions about the knowledge base itself ──────────────────────────
# Questions like "how many contracts do we have" or "what documents are
# loaded" are NOT a fact inside any contract page or SQL table — they're a
# fact about the system's own inventory. Routing these through the normal
# SQL guesser was actively dangerous: with no real table match, it fell back
# to an unrelated table (shipments) and confidently reported the wrong
# number (303, a delivery-document count, presented as "303 contracts").
# Answering this directly from ChromaDB's real document list sidesteps both
# the SQL guesser and Ollama entirely — fast, and can't be wrong.
#
# FIRST VERSION BUG (found by live testing): the old detector required an
# exact phrase like "what contracts" or "what carriers" as a literal
# substring. Real phrasing almost never matches that — "what are the
# DIFFERENT types of contracts" and "what are the different CARRIERS that
# have contract with us" both insert words between the trigger and the
# subject, so the phrase-match never fired and both questions fell through
# to weak semantic search, which is exactly what produced the hallucinated
# "your own company's name is a carrier" answer. Replaced with a
# trigger-word + subject-word check that tolerates words in between.
META_TRIGGER_WORDS = [
    "how many", "what", "which", "list", "name", "show", "types of",
]
META_SUBJECT_WORDS = [
    "contract", "document", "carrier", "pdf", "file",
]
# Guard against false positives: a real content question like "what is the
# termination period for the Vendor A contract" also contains "what" and
# "contract" — but it's asking about a SPECIFIC clause, not the inventory.
# If the question mentions any of these clause-level terms, treat it as a
# content question, not a meta question, even if it also matches the
# trigger/subject words above.
META_EXCLUDE_KEYWORDS = [
    "governing law", "termination", "liability", "insurance", "claim",
    "payment", "rate", "zone", "indemnif", "cargo", "notice period",
    "invoice", "insurance minimum",
]


# ── Capability questions ("what can you help me with?") ────────────────────
# Caught live: a generic capability question isn't a fact in any contract or
# SQL table, so it fell into normal semantic search and got 5 near-random
# chunks back — similarity scores around 0.16-0.17, roughly half the
# strength of even the weak matches seen elsewhere, which is really the
# embedding model saying "none of these are actually relevant." Left alone,
# that also means one wasted Ollama call per capability question, since the
# LLM was being asked to answer from context that had nothing real in it.
# Answering this deterministically avoids both problems.
CAPABILITY_PHRASES = [
    "what can you help", "what can you do", "how can you help",
    "how do you work", "what do you do", "help me with", "what is this",
    "who are you", "what are you",
]


def _is_capability_question(question: str) -> bool:
    q_lower = question.lower().strip()
    if q_lower in ("hi", "hello", "hey", "help", "?"):
        return True
    return any(ask._contains_keyword(q_lower, phrase) for phrase in CAPABILITY_PHRASES)


def _answer_capability_question() -> dict:
    index = rag3.load_index()
    carrier_names = sorted({carrier_display_name(entry) for entry in index.get("documents", {}).values()})
    carrier_list = ", ".join(carrier_names) if carrier_names else "no carriers loaded yet"
    answer = (
        "I can help with two kinds of questions, both answered entirely from data on this "
        "laptop — nothing is sent to the cloud:\n\n"
        f"1. Carrier contract questions — ask about termination, liability caps, insurance "
        f"minimums, governing law, claim deadlines, and similar terms for any of the loaded "
        f"carriers: {carrier_list}. If you don't name a specific carrier and the question could "
        f"mean any of them, I'll ask which one you mean instead of guessing.\n\n"
        "2. Data/rate table questions — ask things like \"what are the rates for New Jersey?\" "
        "and I'll query the structured tables directly.\n\n"
        "You can also ask \"what documents do you have\" or \"how many contracts are loaded\" "
        "to see the current inventory."
    )
    return {"answer": answer}


META_PROXIMITY_WINDOW = 5  # words after a trigger where the subject must appear


def _is_meta_question(question: str) -> bool:
    """
    BUG THIS FIXES (caught live): "what are the performance standards
    required from the carriers?" was wrongly answered as an inventory
    listing of all 12 documents. The old check was `has_trigger and
    has_subject` — true if "what" and "carrier" BOTH appear ANYWHERE in the
    sentence, no matter how far apart or what role they actually play. That
    also meant every new legitimate clause topic ("performance standards",
    "equipment obligations", etc.) needed to be manually added to an
    ever-growing exclude list, which doesn't scale.

    Real inventory questions phrase the subject close to the trigger word:
    "how many CONTRACTS", "what DOCUMENTS do you have", "which CARRIERS".
    This question puts "carriers" eight words after "what", as the object
    of an unrelated clause — so instead of matching anywhere in the
    sentence, require the subject word to appear within a small window
    right after the trigger. This is a more general fix than trying to
    enumerate every possible clause topic.
    """
    q_lower = question.lower()

    # A question that clearly names one specific carrier is asking about
    # THAT carrier's content, not the inventory — even if it also uses a
    # word like "what" or "contract" elsewhere in the sentence.
    if rag3.detect_carrier_filter(question) is not None:
        return False

    if any(ask._contains_keyword(q_lower, kw) for kw in META_EXCLUDE_KEYWORDS):
        return False

    words = q_lower.replace("?", "").replace(",", "").split()
    for trigger in META_TRIGGER_WORDS:
        trigger_words = trigger.split()
        span = len(trigger_words)
        for i in range(len(words) - span + 1):
            if words[i:i + span] != trigger_words:
                continue
            window = words[i + span : i + span + META_PROXIMITY_WINDOW]
            if any(w.startswith(subj) for w in window for subj in META_SUBJECT_WORDS):
                return True
    return False


def _answer_meta_question() -> dict:
    collection = rag3.get_collection()
    docs = rag3.list_documents(collection)
    if not docs:
        return {"answer": "No contracts are currently loaded in the knowledge base.", "documents": []}
    lines = "\n".join(f"- {d['name']}" for d in docs)
    answer = f"There are {len(docs)} contract(s) loaded in the knowledge base:\n\n{lines}"
    return {"answer": answer, "documents": docs}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick check: is the server alive, how big is the knowledge base, how long has it been up."""
    collection    = rag3.get_collection()
    uptime_secs   = int(time.time() - SERVER_START_TIME)
    return {
        "status": "ok",
        "total_chunks": collection.count(),
        "uptime_seconds": uptime_secs,
    }


@app.post("/ask/contracts")
def ask_contracts(req: QuestionRequest):
    """Ask a question about your ingested documents."""
    result = _answer_contract_question(req.question, req.top_k, req.history)
    if result.get("needs_clarification"):
        return {"route": "clarify", **result}
    return result


@app.post("/ask/data")
def ask_data(req: DataQuestionRequest):
    """Ask a plain-English question about a SQL table (e.g. vendor_rates, shipments)."""
    return _answer_data_question(req.table, req.question, req.history)


@app.post("/ask")
def ask_auto(req: AutoQuestionRequest):
    """
    The endpoint the chat UI actually calls. Auto-decides, using the same
    heuristic as ask.py, whether this question needs semantic contract
    search or a structured SQL lookup — so the person asking never has to
    know or care which "mode" they're in. `history` (recent Q&A turns from
    this browser session) is passed through so follow-up questions like
    "what about <a specific document name>?" can be understood in context.
    """
    # Check capability questions ("what can you help me with?") and then
    # meta-questions about the knowledge base's own inventory FIRST, before
    # the normal SQL-vs-contracts routing — neither is answerable by the
    # contract text or the SQL tables, and letting real retrieval have a
    # shot at them was producing near-random low-similarity matches (or, for
    # the SQL guesser, confidently wrong answers).
    if _is_capability_question(req.question):
        result = _answer_capability_question()
        return {"route": "capability", **result}

    if _is_meta_question(req.question):
        result = _answer_meta_question()
        return {"route": "meta", **result}

    # Route on the HISTORY-FOLDED text, not the bare current message. This is
    # the fix for the "and what are those?" bug — that follow-up alone has
    # no data keyword like "zone" or "rate", so it was defaulting to
    # contract search even when the prior turn was clearly a SQL/data
    # question about zones. Folding in the prior question lets the
    # inherited keyword correctly keep it on the data route.
    routing_text = _fold_in_history(req.question, req.history)
    route = ask.route_question(routing_text)
    if route == "data":
        table  = ask.guess_table(routing_text, req.table)
        result = _answer_data_question(table, req.question, req.history)
        return {"route": "data", "table": table, **result}
    else:
        result = _answer_contract_question(req.question, history=req.history)
        if result.get("needs_clarification"):
            return {"route": "clarify", **result}
        return {"route": "contracts", **result}


@app.get("/")
def chat_ui():
    """Serves the chat webpage — this is what teammates actually open in their browser."""
    return FileResponse(Path(__file__).parent / "chat.html")


@app.get("/admin")
def admin_ui():
    """Serves the upload page — for the 2-3 'owner' laptops only, not the whole team."""
    return FileResponse(Path(__file__).parent / "admin.html")


PROJECT_DIR = Path(__file__).parent


@app.post("/admin/ingest")
async def admin_ingest(file: UploadFile = File(...), admin: bool = Depends(require_admin_key)):
    """
    Upload a new (or replacement) contract PDF/docx/etc. Saves it into the
    project folder, then runs the exact same ingestion pipeline as
    `python rag3.py ingest` — chunking, embedding, adding to ChromaDB, and
    auto-generating its index summary. Protected by the admin key (task #14)
    since this WRITES to the shared knowledge base, unlike every /ask
    endpoint, which is read-only and open to anyone on the network.
    """
    dest_path = PROJECT_DIR / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # allow_update=True: re-uploading a file with the same name acts like a
    # refresh (the quarterly-contract-swap case) instead of a hard error.
    result = rag3.ingest_document(str(dest_path), allow_update=True)
    return result
