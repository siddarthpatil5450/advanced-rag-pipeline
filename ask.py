#!/usr/bin/env python3
"""
ask.py — Smart Router (v2)

Automatically decides whether your question needs:
  - STRUCTURED data  -> ask_db.py  (exact numbers: counts, rates, totals, lookups)
  - UNSTRUCTURED text -> rag3.py    (semantic search: contract terms, "what does X do")

Usage:
  python ask.py "what does the termination clause say?"
  python ask.py "how many orders shipped in September 2026?"
  python ask.py "what is the rate for New Jersey?"
  python ask.py "what is the rate for New Jersey?" --table vendor_rates

CUSTOMIZE: the DATA_KEYWORDS, TABLE_HINTS, and DEFAULT_TABLE below are tuned
for a logistics use case (rates, zones, shipments) as a worked example.
Replace them with whatever vocabulary and table names match YOUR data.
"""

import sys
import subprocess


def _contains_keyword(text, keyword):
    """
    Match a keyword at the START of a word, not buried mid-word — but allow
    anything to follow it (so plurals/suffixes like 'rate' -> 'rates' still
    match). Plain 'keyword in text' is unsafe for short generic words like
    'min' or 'max': 'min' is a substring of 'termination', which was
    silently misrouting termination-clause questions to the SQL rate table
    instead of the contract search. But requiring a clean boundary on BOTH
    sides is too strict — it breaks 'rate' matching inside 'rates', which
    caused a different misroute (falling through to the wrong table).
    Checking only the left boundary fixes both: 'min' inside 'ter[min]ation'
    is preceded by a letter, so it's correctly rejected; 'rate' inside
    '[rate]s' is preceded by a space, so it's correctly accepted.
    """
    idx = text.find(keyword)
    while idx != -1:
        before_ok = idx == 0 or not text[idx - 1].isalnum()
        if before_ok:
            return True
        idx = text.find(keyword, idx + 1)
    return False

# Keywords that indicate a STRUCTURED data question -> ask_db.py
DATA_KEYWORDS = [
    "how many", "count", "total", "sum", "average", "avg",
    "max", "min", "highest", "lowest", "most", "least",
    "delivery date", "container number", "september", "october",
    "august", "november", "december", "january", "february",
    "march", "april", "june", "july", "2026", "2025",
    "show rows", "list rows", "filter", "where", "columns",
    "container number", "give container", "show container", "list container",
    # rate / pricing terms — these must never fall through to semantic search,
    # because rag3.py's chunks of a rate table are messy/unreliable for exact numbers.
    # CUSTOMIZE: these are generic freight-industry examples; swap in whatever
    # vocabulary actually matches YOUR structured data's column names.
    "rate", "rates", "zone", "zip code", "postal code",
    "pricing", "price", "accessorial", "min for", "min rate",
]

# ============================================================================
# >>> CUSTOMIZE HERE: map each of your real SQL table names to the words a
# question about it would likely contain. The keys here MUST match table
# names that actually exist in your database (see ask_db.py).
# ============================================================================
TABLE_HINTS = {
    "vendor_rates": ["vendor a", "vendor b"],  # <-- replace with your vendor/carrier names
    "shipments":    ["shipment", "delivery date", "order", "customs"],
}

DEFAULT_TABLE = "shipments"  # <-- CUSTOMIZE: which table to guess when nothing else matches


def route_question(question):
    q_lower = question.lower()
    for keyword in DATA_KEYWORDS:
        if _contains_keyword(q_lower, keyword):
            return "data"
    return "text"


def guess_table(question, override=None):
    if override:
        return override
    q_lower = question.lower()
    for table, hints in TABLE_HINTS.items():
        if any(_contains_keyword(q_lower, h) for h in hints):
            return table
    # If it's clearly a rate/pricing question but no vendor named, guess the
    # only rate table you currently have. CUSTOMIZE: update the table name
    # and rate_words below to match your own schema as you add tables.
    rate_words = ["rate", "min", "zone", "zip code", "postal code", "pricing", "accessorial"]
    if any(_contains_keyword(q_lower, w) for w in rate_words):
        return "vendor_rates"  # <-- CUSTOMIZE: your real rate/pricing table name
    return DEFAULT_TABLE


def ask_text(question):
    print("[ Routing to: rag3.py (semantic search) ]\n")
    subprocess.run(["python", "rag3.py", "query"] + question.split(), capture_output=False)


def ask_data(question, table):
    print(f"[ Routing to: ask_db.py, table = '{table}' ]\n")
    subprocess.run(["python", "ask_db.py", "query", table] + question.split(), capture_output=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"your question\"")
        print("       python ask.py \"your question\" --table vendor_rates")
        return

    args  = sys.argv[1:]
    table_override = None
    if "--table" in args:
        idx = args.index("--table")
        table_override = args[idx + 1]
        args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

    question = " ".join(args)
    route    = route_question(question)

    print(f"\nQuestion : {question}")
    print(f"Route    : {'Structured data (ask_db.py)' if route == 'data' else 'Semantic text (rag3.py)'}")
    print("-" * 60)

    if route == "data":
        table = guess_table(question, table_override)
        ask_data(question, table)
    else:
        ask_text(question)


if __name__ == "__main__":
    main()
