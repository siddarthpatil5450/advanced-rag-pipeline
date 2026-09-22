import os

# ============================================================================
# If you're behind a corporate proxy that does SSL/TLS inspection (re-signs
# certificates), Python's default cert verification can fail on outbound
# calls. DO NOT disable certificate verification globally by default — that
# silently turns off TLS protection for every HTTPS request this process
# makes, which is a real security risk, not just a style issue. Only enable
# this explicitly, and only if you've confirmed you actually need it:
#
#   ALLOW_INSECURE_SSL=1 python rag3.py ingest ...
#
if os.environ.get("ALLOW_INSECURE_SSL") == "1":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
# ============================================================================

import re
import json
import argparse
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
CHROMA_DIR   = Path(__file__).parent / "rag_data3"
COLLECTION   = "knowledge_base"
INDEX_FILE   = CHROMA_DIR / "contract_index.json"  # auto-generated summary of
                                                    # what each ingested document
                                                    # actually covers — this is
                                                    # what the LLM router reads to
                                                    # decide where a question
                                                    # should go, instead of a
                                                    # hand-maintained keyword list
_MODEL_CACHE = Path(__file__).parent / "model_cache" / "all-MiniLM-L6-v2"
EMBED_MODEL  = str(_MODEL_CACHE) if _MODEL_CACHE.exists() else "all-MiniLM-L6-v2"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
CHUNK_SIZE    = 1500  # characters — bigger chunks give the LLM real surrounding
                       # context instead of an isolated sentence, at the cost
                       # of slightly less precise retrieval targeting
CHUNK_OVERLAP = 200
TOP_K         = 5
OLLAMA_NUM_PREDICT = 1024  # max tokens the model is allowed to generate
OLLAMA_TEMPERATURE = 0.4   # a bit of room for elaboration, still fairly factual

# >>> CUSTOMIZE HERE: describe who's asking and what kind of documents this
# is for — this text is folded directly into the LLM's system prompt in
# query_ollama() below, so make it specific to your own team/use case.
ORG_LABEL = "a team member"                     # e.g. "a contracts analyst at Acme Corp"
DOMAIN_LABEL = "internal documents and data"    # e.g. "vendor contracts and shipment data"


# ── Document Loaders ─────────────────────────────────────────────────────────
OCR_MIN_CHARS = 100  # below this, treat the page as image-only and fall back to OCR
OCR_CACHE_DIR = Path(__file__).parent / "rag_data3" / "ocr_cache"


def get_ocr_cache_path(pdf_path, page_index):
    """
    Checkpoint file for one OCR'd page. Naming it after the document + page
    number means: if ingestion gets interrupted (laptop sleep, crash, closed
    terminal), the next run can skip pages we already paid the OCR cost for.
    """
    doc_name  = Path(pdf_path).stem
    safe_name = re.sub(r"[^0-9a-zA-Z_-]", "_", doc_name)
    cache_dir = OCR_CACHE_DIR / safe_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"page_{page_index + 1:04d}.txt"


def load_pdf(path):
    """
    Returns a list of (page_number, text) tuples — page_number is 1-based.
    Keeping pages separate (instead of joining into one big string) means
    every chunk we create later can be traced back to an exact page, which
    is what lets us cite sources like "Page 16" instead of just a filename.
    """
    import pdfplumber

    pages      = []
    ocr_reader = None   # only created if we actually hit a page that needs it
    ocr_doc    = None   # pymupdf handle, only opened if needed

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            if len(text.strip()) >= OCR_MIN_CHARS:
                pages.append((i + 1, text))
                continue

            # Page has little/no real text — likely a scanned/flattened image.
            cache_path = get_ocr_cache_path(path, i)

            if cache_path.exists():
                # Already OCR'd in a previous (interrupted) run — reuse it,
                # no need to pay the slow OCR cost again.
                cached_text = cache_path.read_text(encoding="utf-8")
                pages.append((i + 1, cached_text))
                print(f"    Page {i+1}/{total_pages}: loaded from cache ({len(cached_text)} chars)")
                continue

            # Fall back to OCR (loaded lazily, only once, only if needed).
            if ocr_reader is None:
                import easyocr
                print("    (page has no text layer — loading OCR model, this happens once)")
                ocr_reader = easyocr.Reader(["en"], gpu=False)
            if ocr_doc is None:
                import fitz
                ocr_doc = fitz.open(path)

            pix      = ocr_doc[i].get_pixmap(matrix=fitz_matrix())
            img_path = str(Path(path).parent / "_ocr_temp_page.png")
            pix.save(img_path)

            results  = ocr_reader.readtext(img_path)
            ocr_text = "\n".join(block[1] for block in results)

            # Checkpoint: save immediately, before moving to the next page.
            cache_path.write_text(ocr_text, encoding="utf-8")

            pages.append((i + 1, ocr_text))
            print(f"    Page {i+1}/{total_pages}: OCR fallback used ({len(ocr_text)} chars) — checkpoint saved")

    return pages


def fitz_matrix():
    import fitz
    zoom = 2  # good balance of OCR accuracy vs speed for many pages
    return fitz.Matrix(zoom, zoom)


def load_docx(path):
    from docx import Document
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_csv_excel(path):
    import pandas as pd
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    header = " | ".join(str(c) for c in df.columns)
    rows   = [" | ".join(str(v) for v in row) for _, row in df.iterrows()]
    return header + "\n" + "\n".join(rows)


def load_web(url):
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
    return "\n".join(lines)


def load_image(path):
    """
    Run OCR directly on a standalone image file — e.g. a phone photo or
    screenshot of a table, a whiteboard, a printed page someone snapped a
    picture of. Same lazy-loaded EasyOCR reader and page-checkpoint cache
    used for scanned PDF pages in load_pdf(), just applied to one image
    instead of one page of a multi-page document.
    """
    cache_path = get_ocr_cache_path(path, 0)
    if cache_path.exists():
        cached_text = cache_path.read_text(encoding="utf-8")
        print(f"    Image loaded from OCR cache ({len(cached_text)} chars)")
        return [(1, cached_text)]

    import easyocr
    print("    (running OCR on standalone image — this happens once per file)")
    reader   = easyocr.Reader(["en"], gpu=False)
    results  = reader.readtext(str(path))
    ocr_text = "\n".join(block[1] for block in results)

    cache_path.write_text(ocr_text, encoding="utf-8")
    print(f"    OCR complete ({len(ocr_text)} chars) — checkpoint saved")
    return [(1, ocr_text)]


def load_document(source):
    """
    Returns (pages, name) where pages is always a list of (page_number, text)
    tuples — for PDFs, page_number is the real page. For document types that
    don't have "pages" (Word, Excel, web, plain text, standalone images), we
    use page_number=1 as a single unit, so downstream code (chunking,
    metadata) can treat every document type the same way.
    """
    if source.startswith(("http://", "https://")):
        return [(1, load_web(source))], source
    path = Path(source)
    ext  = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(str(path)), path.name
    elif ext == ".docx":
        return [(1, load_docx(str(path)))], path.name
    elif ext in (".csv", ".xlsx", ".xls"):
        return [(1, load_csv_excel(str(path)))], path.name
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        return load_image(str(path)), path.name
    else:
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))], path.name


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        for sep in ["\n\n", ". ", "? ", "! "]:
            pos = text.rfind(sep, start + chunk_size // 2, end)
            if pos != -1:
                end = pos + len(sep)
                break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return [c for c in chunks if len(c) > 40]


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Chunk each page SEPARATELY instead of joining the whole document into
    one string first. This guarantees a chunk never blends text from two
    different pages together, and lets us tag every chunk with the exact
    page it came from (needed for source citations / grounding).

    Returns a list of (page_number, chunk_text) tuples.
    """
    tagged_chunks = []
    for page_number, text in pages:
        if not text or not text.strip():
            continue
        for chunk in chunk_text(text, chunk_size, overlap):
            tagged_chunks.append((page_number, chunk))
    return tagged_chunks


# ── ChromaDB + Embeddings ────────────────────────────────────────────────────
def get_embedder():
    """Load the sentence-transformer model (cached after first load)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def get_collection():
    """Open (or create) the ChromaDB collection."""
    import chromadb
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )


def list_documents(collection):
    """Return list of unique document metadata stored in ChromaDB."""
    result = collection.get(include=["metadatas"])
    seen   = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "")
        if src not in seen:
            seen[src] = {"source": src, "name": meta.get("name", src),
                         "added": meta.get("added", "")}
    return list(seen.values())


# ── Auto-generated contract index (for the LLM router) ──────────────────────
def load_index():
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return {"documents": {}}


def save_index(index):
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def build_page_excerpts(pages, total_budget=6000):
    """
    Sample a bit of every page (not just the first few pages), spending a
    fixed total character budget spread evenly across all pages. This
    matters because early pages of a contract tend to be general recitals,
    while termination/insurance/rate-schedule content can show up anywhere
    — sampling only the start of the document would bias the summary
    towards whatever happens to be on page 1-2.
    """
    if not pages:
        return ""
    per_page_budget = max(50, total_budget // len(pages))
    lines = []
    for page_num, text in pages:
        snippet = (text or "").strip().replace("\n", " ")[:per_page_budget]
        if snippet:
            lines.append(f"[Page {page_num}] {snippet}")
    return "\n".join(lines)


def _ollama_generate_with_retry(prompt, model=OLLAMA_MODEL, num_predict=350,
                                  temperature=0.3, timeout=400, max_attempts=2):
    """
    Shared helper for the ingest-time Ollama calls (document summary, key
    terms extraction). This is the real fix for the Vendor A failure — not just
    a bigger timeout. The FIRST call after Ollama has been idle pays a real
    cold-load cost on top of whatever the prompt itself needs, and that
    combination alone blew past even a generous timeout. Retrying
    immediately hits an Ollama that's already mid-load or fully warm from
    the first attempt, and usually succeeds where the first one timed out
    — this matters a lot more now, ingesting a batch of 12 documents where
    any one of them could be first in line after an idle period.
    """
    import requests
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"num_predict": num_predict, "temperature": temperature},
                },
                timeout=timeout
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip(), None
            last_error = f"Ollama HTTP error {resp.status_code}"
        except Exception as e:
            last_error = str(e)
        if attempt < max_attempts:
            print(f"    (attempt {attempt} failed: {last_error} — retrying...)")
    return None, last_error


def generate_document_summary(pages, name, model=OLLAMA_MODEL):
    """
    Ask the LOCAL Ollama model (never a cloud service) to write a short
    index entry describing what this document actually covers, based on
    excerpts sampled across every page. This is what lets the router
    generalize to new phrasings instead of needing a hand-maintained
    keyword list updated every time something gets misrouted.
    """
    excerpts = build_page_excerpts(pages)
    prompt = (
        "You are writing a short index entry for a document search system. "
        "Based on the page excerpts below (each labeled with its real page "
        "number), write a concise 3-5 sentence summary of what topics, "
        "clauses, or sections this specific document covers. Be specific "
        "about the KINDS of questions this document could answer (e.g. "
        "termination terms, insurance/liability requirements, payment or "
        "rate schedules, equipment obligations, service scope) and mention "
        "roughly which page numbers key topics appear on, if you can tell. "
        "Do not add generic filler like 'this is a legal document' — be "
        "specific to what's actually in THIS text.\n\n"
        f"Document name: {name}\n"
        f"Total pages: {len(pages)}\n\n"
        f"PAGE EXCERPTS:\n{excerpts}\n\n"
        "SUMMARY:"
    )
    result, error = _ollama_generate_with_retry(prompt, model=model, num_predict=350, temperature=0.3)
    return result if result is not None else f"[Could not auto-summarize — {error}]"


# Specific, high-stakes facts worth extracting as a structured table rather
# than leaving buried in a vague summary — modeled on the "Key Terms" table
# pattern from a full contract-review skill (governing law, liability
# direction/cap, claim deadlines, etc.). A narrow, specific extraction like
# this is something a small local model handles far more reliably than
# open-ended summarization, because there's a defined right answer per field
# instead of an unbounded "write something reasonable" task.
#
# >>> CUSTOMIZE HERE: this example list is tuned for LTL freight contracts.
# Replace these fields with whatever high-stakes facts matter in YOUR
# documents (e.g. renewal date, SLA thresholds, data retention period).
KEY_TERMS_FIELDS = [
    "Governing Law",
    "Termination Notice Period",
    "Q4 Termination Restriction (does the counterparty need our consent to terminate in Q4?)",
    "Liability Direction (is it the GREATER of two values, or the LESSER?)",
    "Liability Cap (dollar amount and/or per-unit rate)",
    "Claim Filing Timebar (deadline to file a claim after delivery)",
    "Claim Rejection Window (deadline for the other party to reject a claim before it's deemed accepted)",
    "Payment Terms (e.g. Net 30)",
    "Insurance Minimums (CGL, Auto, Umbrella, Cargo)",
]


def generate_key_terms(pages, name, model=OLLAMA_MODEL):
    """
    Extract specific, named facts into a structured table, instead of a
    free-form summary. IMPORTANT: this is generated by a small local model
    reading sampled excerpts, NOT verified by a human — treat every value as
    a lead to confirm against the real document, not a certified fact. The
    caller stores this with a `key_terms_verified: False` flag for exactly
    this reason; flip it to True only after someone has actually checked it
    against the source PDF.
    """
    excerpts    = build_page_excerpts(pages, total_budget=8000)
    fields_list = "\n".join(f"- {f}" for f in KEY_TERMS_FIELDS)
    prompt = (
        "You are extracting specific facts from a contract into a reference table. "
        "Based on the page excerpts below (each labeled with its real page number), "
        "find the value for EACH field listed. For each one, output exactly one line:\n"
        "FIELD NAME: value [Page X]\n\n"
        "If a field is not shown in the excerpts below, write exactly:\n"
        "FIELD NAME: Not found in sampled pages\n\n"
        "Do not guess, do not use outside knowledge about typical contracts — only "
        "state what is actually shown in the excerpts. Do not skip any field.\n\n"
        f"FIELDS TO FIND:\n{fields_list}\n\n"
        f"Document name: {name}\n\n"
        f"PAGE EXCERPTS:\n{excerpts}\n\n"
        "EXTRACTED TABLE:"
    )
    # temperature kept low (0.1) — this is factual extraction, not creative writing
    result, error = _ollama_generate_with_retry(prompt, model=model, num_predict=500, temperature=0.1)
    return result if result is not None else f"[Could not extract key terms — {error}]"


def update_index_for_document(source, name, pages):
    """Called automatically after a successful ingest/update — regenerates
    this one document's index entry so it never goes stale by hand."""
    print(f"    Generating index summary for '{name}' (one-time per ingest)...")
    summary = generate_document_summary(pages, name)
    print(f"    Extracting key terms table for '{name}'...")
    key_terms = generate_key_terms(pages, name)
    index = load_index()
    index["documents"][source] = {
        "name": name,
        "summary": summary,
        "key_terms": key_terms,
        "key_terms_verified": False,  # flip to True by hand once a human has
                                       # actually checked this against the PDF
        "total_pages": len(pages),
        "updated": datetime.now().isoformat(),
    }
    save_index(index)
    print(f"    Index entry updated for '{name}'.")


def remove_from_index(source):
    """Called automatically on delete, so the index doesn't reference a
    document that no longer exists in the knowledge base."""
    index = load_index()
    if source in index.get("documents", {}):
        del index["documents"][source]
        save_index(index)


# ── Carrier/vendor-name metadata filtering (task #5) ─────────────────────────
# As the knowledge base grows past a handful of documents, similar
# boilerplate across counterparties (insurance minimums, termination clauses)
# makes it easier for retrieval to accidentally pull in the wrong document's
# chunk alongside the right one. If a question clearly names ONE counterparty,
# we can sidestep that risk entirely by restricting the ChromaDB query to
# only that document's chunks — rather than hoping semantic similarity
# naturally keeps them separate.
#
# >>> CUSTOMIZE HERE: add your OWN organization's name and any other
# boilerplate words that appear in every one of your filenames (so they
# don't get mistaken for a counterparty's name). The words below are a
# generic starting point for contract-style filenames.
_CARRIER_NAME_STOPWORDS = {
    "executed", "complete", "docusign", "master", "direct", "transportation",
    "agreement", "contract", "express", "freight", "lines", "logistics",
    "services", "company", "llc", "inc", "transport", "carrier", "global",
    "with", "the", "for", "and", "pdf", "corp", "corporation", "line",
    "service", "group", "systems", "delivery", "mtsa", "tsa", "osa", "lsa",
    "clean", "final",
    # "yourcompany", "yourteam",  # <-- add your own org's name here
}


def _mentions_whole_word(text, word):
    """
    Whole-word substring check — same reasoning as ask.py's
    _contains_keyword: prevents e.g. 'Old' matching inside some unrelated
    longer word, while still matching plurals/suffixes naturally. Kept as a
    small local copy here rather than importing ask.py, to avoid any risk
    of a circular import between the two modules.
    """
    text = text.lower()
    word = word.lower()
    idx = text.find(word)
    while idx != -1:
        before_ok = idx == 0 or not text[idx - 1].isalnum()
        if before_ok:
            return True
        idx = text.find(word, idx + 1)
    return False


def extract_carrier_tokens(doc_name):
    """
    Pull the significant, identifying words out of a document's filename —
    skipping generic contract boilerplate — so we can later check whether a
    question is naming this specific carrier/vendor. E.g. from
    '(Executed) Acme_Corp_Master Direct Transportation Agreement.pdf'
    this extracts just ['Acme', 'Corp'].
    """
    base  = Path(doc_name).stem
    words = re.findall(r"[A-Za-z]+", base)
    return [w for w in words if w.lower() not in _CARRIER_NAME_STOPWORDS and len(w) >= 3]


def detect_carrier_filter(question, index=None):
    """
    If the question clearly names exactly one ingested carrier, return that
    document's source key so retrieval can be restricted to ONLY that
    carrier's chunks. Returns None for general questions, or ones that
    don't clearly point at a single carrier (including ties between two
    equally-plausible carriers) — those still search across everything,
    same as before. Deliberately conservative: better to fall back to
    unfiltered search than to risk excluding the actually-correct document.
    """
    index   = index if index is not None else load_index()
    matches = []
    for source, entry in index.get("documents", {}).items():
        tokens = extract_carrier_tokens(entry.get("name", source))
        if not tokens:
            continue
        hits = sum(1 for t in tokens if _mentions_whole_word(question, t))
        if hits > 0:
            matches.append((hits, source))

    if not matches:
        return None
    matches.sort(reverse=True)
    if len(matches) == 1 or matches[0][0] > matches[1][0]:
        return matches[0][1]
    return None  # ambiguous — don't risk filtering out the right document


# ── Ollama ───────────────────────────────────────────────────────────────────
def format_history(history, max_turns=4):
    """
    Turn a list of {"question": ..., "answer": ...} dicts (most recent last)
    into a short text block for the prompt. Only the last few turns are kept
    — sending the whole conversation would blow up the prompt size and slow
    an already-slow CPU model down further, and old turns are rarely still
    relevant to a follow-up question anyway.
    """
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if not q:
            continue
        # Keep each prior answer short in the prompt — we only need enough
        # of it to resolve "it"/"that"/"what about X" style references, not
        # the full three-part answer again.
        a_short = a[:300] + ("..." if len(a) > 300 else "")
        lines.append(f"Q: {q}\nA: {a_short}")
    return "\n\n".join(lines)


def query_ollama(question, context, model=OLLAMA_MODEL, history=None):
    import requests
    history_text = format_history(history)
    history_block = (
        f"PRIOR CONVERSATION (for reference only — use it to understand what "
        f"'it', 'that', or a follow-up like 'what about X' refers to, but do "
        f"NOT treat it as a source of facts; only the CONTEXT below is a "
        f"valid source of facts):\n{history_text}\n\n"
        if history_text else ""
    )
    prompt = (
        f"You are an analyst helping {ORG_LABEL} understand {DOMAIN_LABEL}. "
        "Use ONLY the context below to answer — do not use outside knowledge.\n\n"
        "Each context block below is labeled with its real source document and page number, "
        "in the exact format '[Source: <the real filename shown in that block>, Page <the "
        "real page number shown in that block>]'. When you cite your source, copy the ACTUAL "
        "filename and page number that appear in the CONTEXT section below — do not write the "
        "literal placeholder words 'filename' or 'Page 16'; those are not real values, they do "
        "not exist in this conversation, and copying them produces a fake citation. If different "
        "documents give different or conflicting answers, mention each one separately by its "
        "real source and page — do not blend them into one answer.\n\n"
        "If the CONTEXT section below does not actually contain information that answers this "
        "specific QUESTION — for example, if the question uses a pronoun like 'it', 'those', or "
        "'that' and the CONTEXT below doesn't contain matching content — say clearly that the "
        "current context doesn't cover this, and suggest the person rephrase with the specific "
        "topic named. Do NOT fall back to repeating an answer from PRIOR CONVERSATION as if it "
        "came from the CONTEXT below; PRIOR CONVERSATION is background only, never a source of "
        "facts.\n\n"
        "CONTRACT ISOLATION RULE: each carrier's contract is a separate, independent "
        "agreement. NEVER apply one carrier's terms, rates, or clauses to a different "
        "carrier, even if the wording looks similar. If a 'Reference key terms' block "
        "appears in the context below, it belongs ONLY to the document named right above "
        "it — do not mix its values into a different carrier's answer.\n\n"
        "Structure your answer in three parts:\n"
        "1. Direct Answer — state the clear answer immediately, in 1-2 sentences.\n"
        "2. Context — explain the relevant clause/section in more detail, quoting or "
        "paraphrasing the specific contract language that supports the answer.\n"
        "3. Practical Note — briefly note anything the reader should be aware of or double-check "
        "(e.g. this only applies to one of the three contracts, or there's a related clause nearby).\n\n"
        "If the answer is not in the context, say so clearly instead of guessing — do not skip "
        "the structure just because the answer is short.\n\n"
        f"{history_block}"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",  # keep the model loaded in memory for 30 min
                                      # after each request, instead of Ollama's default
                                      # 5 min — avoids a slow "cold reload" every time
                                      # someone queries after a short gap
                "options": {
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "temperature": OLLAMA_TEMPERATURE,
                }
            },
            timeout=240  # generous — covers a full cold model load plus generation
                         # on CPU, which the previous 120s cut off mid-response
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"[Ollama HTTP error {resp.status_code}]"
    except Exception as e:
        if "Connection" in type(e).__name__:
            return "[Cannot reach Ollama — run: ollama serve]"
        return f"[Error: {e}]"


# ── Commands ─────────────────────────────────────────────────────────────────
def ingest_document(source, embedder=None, collection=None, allow_update=False):
    """
    The actual work of ingesting ONE document — pulled out of cmd_ingest so
    the CLI and the server's upload endpoint share identical logic instead
    of two copies that can drift apart. Returns a plain dict describing what
    happened, so callers can print it (CLI) or return it as JSON (server).

    allow_update=True lets this overwrite an already-ingested document's
    chunks instead of refusing — that's what the server's upload endpoint
    uses so re-uploading the same filename acts like a refresh, not an error.
    """
    embedder   = embedder or get_embedder()
    collection = collection or get_collection()

    try:
        pages, name = load_document(source)
        has_text = any(text.strip() for _, text in pages)
        if not has_text:
            return {"success": False, "name": name, "error": "No text found in document."}

        existing = collection.get(where={"source": source})
        if existing["ids"]:
            if not allow_update:
                return {
                    "success": False, "name": name,
                    "error": f"'{name}' already ingested. Use 'update' to refresh."
                }
            collection.delete(ids=existing["ids"])

        tagged_chunks = chunk_pages(pages)
        chunks        = [c for _, c in tagged_chunks]
        page_numbers  = [p for p, _ in tagged_chunks]
        embeddings    = embedder.encode(chunks, show_progress_bar=True).tolist()
        added_at      = datetime.now().isoformat()

        ids       = [f"{source}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "name": name, "page": page_numbers[i],
                      "chunk_id": i, "added": added_at}
                     for i in range(len(chunks))]

        # ChromaDB has a batch limit — add in batches of 500
        batch = 500
        for start in range(0, len(chunks), batch):
            collection.add(
                ids        = ids[start:start+batch],
                documents  = chunks[start:start+batch],
                embeddings = embeddings[start:start+batch],
                metadatas  = metadatas[start:start+batch],
            )

        update_index_for_document(source, name, pages)
        return {"success": True, "name": name, "chunks": len(chunks), "pages": len(pages)}
    except Exception as e:
        return {"success": False, "name": Path(source).name, "error": str(e)}


def cmd_ingest(args):
    embedder   = get_embedder()
    collection = get_collection()

    for source in args.sources:
        print(f"\nIngesting: {source}")
        result = ingest_document(source, embedder, collection)
        if result["success"]:
            print(f"  OK  {result['chunks']} chunks indexed from '{result['name']}'")
        else:
            print(f"  FAIL: {result['error']}")

    total = collection.count()
    docs  = list_documents(collection)
    print(f"\nKnowledge base: {total} chunks from {len(docs)} document(s).")


def cmd_query(args):
    collection = get_collection()
    if collection.count() == 0:
        print("Knowledge base is empty. Run:  python rag3.py ingest <file>")
        return

    question = " ".join(args.question)
    print(f"\nQuestion : {question}")

    # Embed the query and search
    embedder  = get_embedder()
    q_vector  = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings = q_vector,
        n_results        = args.top_k,
        include          = ["documents", "metadatas", "distances"]
    )

    docs_found = results["documents"][0]
    metas      = results["metadatas"][0]
    distances  = results["distances"][0]

    if not docs_found:
        print("No relevant chunks found.")
        return

    print(f"\n-- Retrieved {len(docs_found)} chunks --")
    context_parts = []
    seen_pages     = set()   # avoid sending the same full page twice

    for i, (doc, meta, dist) in enumerate(zip(docs_found, metas, distances), 1):
        score = round(1 - dist, 3)   # cosine similarity (1 = perfect match)
        page  = meta.get("page")
        source = meta.get("source")
        name   = meta.get("name", "?")
        page_label = f", Page {page}" if page else ""
        print(f"  [{i}] similarity={score}  source={name}{page_label}")

        page_key = (source, page)
        if page_key in seen_pages:
            continue  # already pulled this whole page in for an earlier chunk
        seen_pages.add(page_key)

        # Parent-page retrieval: instead of sending just this one small
        # chunk, pull EVERY chunk from the same page and stitch them back
        # together — the LLM gets full surrounding context, not an
        # isolated fragment, while search itself still matched precisely.
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

    context = "\n\n---\n\n".join(context_parts)
    model   = args.model or OLLAMA_MODEL
    print(f"\nGenerating answer with {model} ...\n")
    answer  = query_ollama(question, context, model)
    print("=" * 60)
    print(answer)
    print("=" * 60)


def cmd_list(args):
    collection = get_collection()
    docs = list_documents(collection)
    if not docs:
        print("Knowledge base is empty.")
        return
    print(f"\n{len(docs)} document(s):")
    for i, d in enumerate(docs):
        print(f"  [{i}] {d['name']:<40s}  added {d['added'][:10]}")
    print(f"\nTotal chunks: {collection.count()}")


def cmd_clear(args):
    confirm = input("Delete all indexed data? [y/N] ")
    if confirm.strip().lower() == "y":
        import shutil
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        print("Cleared.")
    else:
        print("Cancelled.")


def cmd_update(args):
    """Remove old chunks for a source and re-ingest the updated file."""
    collection = get_collection()
    source     = args.source
    path       = Path(source)
    name       = source if source.startswith(("http://", "https://")) else path.name

    # Find existing chunks for this source
    existing = collection.get(where={"source": source})
    if not existing["ids"]:
        # Try matching by name
        existing = collection.get(where={"name": name})
    if not existing["ids"]:
        print(f"'{name}' not found in knowledge base. Use ingest instead.")
        return

    old_count = len(existing["ids"])
    collection.delete(ids=existing["ids"])
    print(f"Removed {old_count} old chunks for '{name}'.")

    # Re-ingest
    print(f"Re-ingesting: {source}")
    try:
        embedder    = get_embedder()
        pages, name = load_document(source)
        has_text    = any(text.strip() for _, text in pages)
        if not has_text:
            print(f"  ! No text found in {name}")
            return

        tagged_chunks = chunk_pages(pages)
        chunks        = [c for _, c in tagged_chunks]
        page_numbers  = [p for p, _ in tagged_chunks]
        embeddings    = embedder.encode(chunks, show_progress_bar=True).tolist()
        added_at      = datetime.now().isoformat()

        ids       = [f"{source}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "name": name, "page": page_numbers[i],
                      "chunk_id": i, "added": added_at}
                     for i in range(len(chunks))]

        batch = 500
        for start in range(0, len(chunks), batch):
            collection.add(
                ids        = ids[start:start+batch],
                documents  = chunks[start:start+batch],
                embeddings = embeddings[start:start+batch],
                metadatas  = metadatas[start:start+batch],
            )
        print(f"  OK  {len(chunks)} new chunks indexed from '{name}'")
        update_index_for_document(source, name, pages)  # regenerates the
                                                          # summary from the
                                                          # NEW file — this is
                                                          # the actual fix for
                                                          # quarterly contract
                                                          # refreshes staying
                                                          # accurate
    except Exception as e:
        print(f"  FAIL: {e}")

    total = collection.count()
    docs  = list_documents(collection)
    print(f"\nKnowledge base: {total} chunks from {len(docs)} document(s).")


def cmd_delete(args):
    """Remove a document and all its chunks from the knowledge base."""
    collection = get_collection()
    source     = args.source
    path       = Path(source)
    name       = source if source.startswith(("http://", "https://")) else path.name

    existing = collection.get(where={"source": source})
    if not existing["ids"]:
        existing = collection.get(where={"name": name})
    if not existing["ids"]:
        print(f"'{name}' not found in knowledge base.")
        print("Run: python rag3.py list")
        return

    confirm = input(f"Delete '{name}' and all its chunks? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Cancelled.")
        return

    count = len(existing["ids"])
    collection.delete(ids=existing["ids"])
    remove_from_index(source)
    print(f"Deleted '{name}' — removed {count} chunks.")
    print(f"Knowledge base: {collection.count()} chunks remaining.")


def cmd_check_index(args):
    """
    Scan the contract index for any document whose auto-generated summary
    or key-terms extraction actually failed (e.g. an Ollama timeout during
    ingest) — this is exactly the gap that happened silently with Vendor A the
    first time. Run this after any ingest/update batch, especially a big
    one, to get a clear checklist of what needs a re-run instead of having
    to manually read through the whole JSON file.
    """
    index = load_index()
    docs  = index.get("documents", {})
    if not docs:
        print("No documents in the index yet.")
        return

    problems = []
    unverified = []
    for source, entry in docs.items():
        summary   = entry.get("summary", "") or ""
        key_terms = entry.get("key_terms", "") or ""
        if summary.startswith("[Could not") or key_terms.startswith("[Could not"):
            problems.append((entry.get("name", source), source))
        elif not entry.get("key_terms_verified", False):
            unverified.append(entry.get("name", source))

    print(f"{len(docs)} document(s) in the index.\n")

    if problems:
        print(f"{len(problems)} document(s) FAILED to auto-summarize/extract — re-run these:")
        for name, source in problems:
            print(f"  - {name}")
            print(f"      python rag3.py update \"{source}\"")
        print()
    else:
        print("No failed summaries or key-terms extractions. Good.\n")

    if unverified:
        print(f"{len(unverified)} document(s) have key terms NOT YET human-verified:")
        for name in unverified:
            print(f"  - {name}")
        print("  (spot-check these against the real PDF, then flip key_terms_verified to true)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline v3 — Vector Search")
    sub    = parser.add_subparsers(dest="cmd")

    p_in = sub.add_parser("ingest", help="Add documents to the knowledge base")
    p_in.add_argument("sources", nargs="+")

    p_q = sub.add_parser("query", help="Ask a question")
    p_q.add_argument("question", nargs="+")
    p_q.add_argument("--top-k", type=int, default=TOP_K)
    p_q.add_argument("--model", default=None)

    sub.add_parser("list",  help="Show all indexed documents")
    sub.add_parser("clear", help="Delete the entire knowledge base")

    p_up = sub.add_parser("update", help="Re-ingest an updated file")
    p_up.add_argument("source", help="File path or URL to update")

    p_del = sub.add_parser("delete", help="Remove a document from the knowledge base")
    p_del.add_argument("source", help="File path or URL to delete")

    sub.add_parser("check-index", help="Report any documents with a failed or unverified index entry")

    args     = parser.parse_args()
    dispatch = {
        "ingest":      cmd_ingest,
        "query":       cmd_query,
        "list":        cmd_list,
        "clear":       cmd_clear,
        "update":      cmd_update,
        "delete":      cmd_delete,
        "check-index": cmd_check_index,
    }
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
