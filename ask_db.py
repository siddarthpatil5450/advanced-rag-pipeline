"""
ask_db.py — SQLite-based structured data engine.

Replaces ask_data.py's per-query Excel loading with a real relational database.
Data is loaded ONCE into SQLite tables, then queried with actual SQL.

Commands:
    register <table_name> <filepath>   Load an Excel/CSV file into a SQL table
    list                                Show all tables and row counts
    schema <table_name>                 Show a table's columns and types
    query <table_name> "<question>"     Ask a question in plain English (keyword-based -> SQL)
    sql "<raw SQL query>"               Run raw SQL directly (power user mode)
"""

import re
import sqlite3
import argparse
from pathlib import Path

DB_FILE = Path(__file__).parent / "logistics.db"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# Tables wider than this get their column list shortlisted (see
# shortlist_relevant_columns) before going into the SQL-generation prompt —
# keeps the prompt small on a CPU-bound local model and reduces the chance
# it references a column that has nothing to do with the actual question.
WIDE_TABLE_THRESHOLD     = 12
MAX_SHORTLISTED_COLUMNS  = 15


# ── Connection ───────────────────────────────────────────────────────────────
def get_connection():
    return sqlite3.connect(str(DB_FILE))


def sanitize_column(name):
    """SQL column names can't have spaces, dots, or special characters."""
    name = str(name).strip()
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if name and name[0].isdigit():
        name = "col_" + name
    return name or "unnamed"


def _format_history_for_sql(history, max_turns=3):
    """
    Short text block of recent Q&A turns, for follow-up questions like "what
    about Zone 4?" that only make sense with the prior question's context.
    Kept as its own small helper (rather than importing rag3.format_history)
    so ask_db.py has no dependency on rag3.py.
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
        a_short = a[:200] + ("..." if len(a) > 200 else "")
        lines.append(f"Q: {q}\nA: {a_short}")
    return "\n\n".join(lines)


def query_ollama_for_sql(question, table, schema_text, sample_text="", previous_attempt=None, history=None):
    """
    Ask the local model to write ONE SQL SELECT statement for this question.

    sample_text: a few real rows from the table, so the model can see the
    ACTUAL formatting of values (e.g. 'NJ' vs 'New Jersey', dates as
    'YYYY-MM-DD' vs 'MM/DD/YYYY') instead of guessing and writing a WHERE
    clause that silently matches zero rows.

    previous_attempt: {"sql": ..., "issue": ...} — set on a retry, so the
    model sees exactly what it tried before and why it didn't work, instead
    of repeating the same mistake blind.

    history: recent conversation turns, so a follow-up like "what about
    Zone 4?" can inherit context from the previous question.
    """
    import requests

    history_text  = _format_history_for_sql(history)
    history_block = (
        "RECENT CONVERSATION (for context on follow-up questions like 'what about "
        "X' only — not a source of data; only the SCHEMA and SAMPLE ROWS below are "
        f"real data):\n{history_text}\n\n" if history_text else ""
    )
    retry_block = (
        f"Your previous attempt was:\n{previous_attempt['sql']}\n"
        f"Problem: {previous_attempt['issue']}\n"
        "Write a corrected query that fixes this.\n\n"
        if previous_attempt else ""
    )
    sample_block = (
        f"SAMPLE ROWS (real data, showing the actual value formats):\n{sample_text}\n\n"
        if sample_text else ""
    )

    prompt = (
        "You are a SQLite expert. Write ONE SQL query that answers the question.\n"
        "Rules:\n"
        "- Only output the raw SQL query, nothing else. No explanation, no markdown, no code fences.\n"
        "- Only write a SELECT statement. Never write UPDATE, DELETE, INSERT, or DROP.\n"
        "- Match the exact formatting of values shown in the SAMPLE ROWS (abbreviations, "
        "capitalization, date format) — do not assume a different format.\n"
        f"- The table is named '{table}' with these columns:\n{schema_text}\n\n"
        f"{sample_block}"
        f"{history_block}"
        f"{retry_block}"
        f"Question: {question}\n\n"
        "SQL query:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 300, "temperature": 0.1},
            },
            timeout=120
        )
        raw_sql = resp.json().get("response", "").strip()
    except Exception as e:
        return f"-- ERROR: could not reach Ollama ({e})"
    raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    return raw_sql



# ── Register: load a file into a SQL table ──────────────────────────────────
def cmd_register(args):
    import pandas as pd

    path = Path(args.filepath)
    if not path.exists():
        print(f"File not found: {args.filepath}")
        return

    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        print(f"Unsupported file type: {ext}")
        return

    # Clean column names so SQL can use them safely
    original_cols = list(df.columns)
    df.columns = [sanitize_column(c) for c in df.columns]

    conn = get_connection()
    df.to_sql(args.table, conn, if_exists="replace", index=False)
    conn.close()

    print(f"Loaded '{path.name}' -> table '{args.table}'")
    print(f"  {len(df)} rows x {len(df.columns)} columns")
    print(f"\n  Columns (original -> SQL-safe name):")
    for orig, safe in zip(original_cols, df.columns):
        marker = "  (renamed)" if orig != safe else ""
        print(f"    {orig}  ->  {safe}{marker}")


# ── List: show all tables ───────────────────────────────────────────────────
def cmd_list(args):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]

    if not tables:
        print("No tables in database yet. Run: python ask_db.py register <table> <file>")
        conn.close()
        return

    print(f"\n{len(tables)} table(s) in {DB_FILE.name}:")
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        count = cur.fetchone()[0]
        cur.execute(f"PRAGMA table_info({t})")
        num_cols = len(cur.fetchall())
        print(f"  [{t}]  {count} rows x {num_cols} columns")
    conn.close()


# ── Schema: show a table's structure ────────────────────────────────────────
def cmd_schema(args):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(f"PRAGMA table_info({args.table})")
    columns = cur.fetchall()
    conn.close()

    if not columns:
        print(f"Table '{args.table}' not found. Run: python ask_db.py list")
        return

    print(f"\nSchema for '{args.table}':")
    for col in columns:
        # col = (index, name, type, notnull, default, is_primary_key)
        print(f"  {col[1]:<30s} {col[2]}")


# ── Raw SQL: power-user escape hatch ────────────────────────────────────────
def cmd_sql(args):
    import pandas as pd
    query      = " ".join(args.query)
    is_select  = query.strip().upper().startswith("SELECT")
    conn       = get_connection()
    try:
        if is_select:
            df = pd.read_sql_query(query, conn)
            print(f"\n{len(df)} row(s):\n")
            print(df.to_string(index=False))
        else:
            # UPDATE / INSERT / DELETE don't return rows — use a plain
            # cursor instead of pandas, and commit so the change is saved.
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()
            print(f"\nOK — {cur.rowcount} row(s) affected.")
    except Exception as e:
        print(f"SQL Error: {e}")
    conn.close()


# ── Query: plain-English question -> SQL ────────────────────────────────────
def get_columns(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def get_numeric_columns(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    # SQLite types: INTEGER, REAL, TEXT, BLOB
    return [row[1] for row in cur.fetchall() if row[2] in ("INTEGER", "REAL")]


def find_matching_columns(columns, words):
    matches = []
    for col in columns:
        if any(w in col.lower() for w in words):
            matches.append(col)
    return matches


def shortlist_relevant_columns(question, columns):
    """
    For a wide table, sending every column name into the SQL-generation
    prompt wastes context budget and can distract a small local model into
    referencing a column that has nothing to do with the question. Score
    each column by whether a significant word from the question appears in
    it, and keep only the matches — falling back to the full (capped) list
    if nothing matches, so a vague question never loses access to a column
    it might actually need.
    """
    q_words = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) >= 3]
    if not q_words:
        return columns[:MAX_SHORTLISTED_COLUMNS]

    scored = []
    for col in columns:
        col_lower = col.lower()
        hits = sum(1 for w in q_words if w in col_lower)
        if hits > 0:
            scored.append((hits, col))

    if not scored:
        return columns[:MAX_SHORTLISTED_COLUMNS]

    scored.sort(reverse=True)
    return [col for _, col in scored[:MAX_SHORTLISTED_COLUMNS]]


def clean_column_label(label):
    """
    Raw SQL returns a column named after the exact expression that produced
    it (e.g. 'AVG(rate)', 'UPPER(carrier)') when the query has no explicit
    AS alias. Left as-is, a natural-language summarizer can misread the SQL
    syntax as if it were literal data. Strips the SQL wrapper and returns a
    short, readable label instead — e.g. 'AVG(rate)' -> 'average rate'.
    """
    if not label:
        return label
    m = re.match(r"^\s*([A-Za-z_]+)\s*\(\s*([A-Za-z0-9_.*]*)\s*\)\s*$", label)
    if not m:
        return label.replace("_", " ")
    func, arg = m.group(1).upper(), m.group(2)
    arg_clean = arg.replace("_", " ") if arg else ""
    FUNC_LABELS = {"AVG": "average", "SUM": "total", "COUNT": "count",
                   "MAX": "maximum", "MIN": "minimum"}
    if func in FUNC_LABELS:
        return f"{FUNC_LABELS[func]} {arg_clean}".strip() if arg_clean else FUNC_LABELS[func]
    # UPPER/LOWER/TRIM/etc. — the wrapper is just formatting, not part of
    # the meaningful label; drop it and keep the underlying column name.
    return arg_clean or label


def clean_row_labels(rows):
    """Apply clean_column_label to every row dict's keys."""
    return [{clean_column_label(k): v for k, v in row.items()} for row in rows]


def summarize_result_in_words(question, rows, model=OLLAMA_MODEL):
    """
    Turn raw SQL result rows into a short natural-language sentence for the
    chat UI, instead of falling back to a raw JSON dump. Expects rows that
    have already been through clean_row_labels, so the model never has to
    parse SQL syntax embedded in a column name as if it were real data.
    """
    if not rows:
        return "No matching rows were found for this question."

    import requests
    # Cap how many rows go into the prompt — a big result set doesn't need
    # every row to summarize accurately, and keeps the prompt small on a
    # CPU-bound local model.
    sample     = rows[:15]
    rows_text  = "\n".join(str(r) for r in sample)
    more_note  = (f"\n(... and {len(rows) - len(sample)} more row(s) not shown above)"
                  if len(rows) > len(sample) else "")
    prompt = (
        "Answer the question below in ONE short, natural sentence, using ONLY the "
        "data rows shown. Do not mention SQL, tables, or column syntax — just state "
        "the answer plainly, the way you'd say it out loud.\n\n"
        f"QUESTION: {question}\n\n"
        f"DATA ROWS:\n{rows_text}{more_note}\n\n"
        "ANSWER:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 200, "temperature": 0.2},
            },
            timeout=120
        )
        if resp.status_code == 200:
            text = resp.json().get("response", "").strip()
            if text:
                return text
    except Exception:
        pass
    # Fallback if Ollama is unreachable or returns nothing — still give a
    # useful answer instead of silently failing.
    if len(rows) == 1:
        return "Result: " + ", ".join(f"{k} = {v}" for k, v in rows[0].items())
    return f"Found {len(rows)} matching row(s)."


def cmd_query(args):
    conn    = get_connection()
    columns = get_columns(conn, args.table)
    if not columns:
        print(f"Table '{args.table}' not found. Run: python ask_db.py list")
        conn.close()
        return

    question = " ".join(args.question).lower()
    words    = question.split()
    print(f"\nTable   : {args.table}")
    print(f"Question: {question}\n")

    cur = conn.cursor()

    # --- DATE FILTER ---
    months = {"january":"01","february":"02","march":"03","april":"04",
              "may":"05","june":"06","july":"07","august":"08",
              "september":"09","october":"10","november":"11","december":"12"}
    month_num = next((num for m, num in months.items() if m in question), None)
    year      = next((w for w in words if w.isdigit() and len(w) == 4), None)

    if month_num or year:
        date_cols = [c for c in columns if any(k in c.lower() for k in ["date", "eta", "etd", "ata"])]
        target_col = next((c for c in date_cols if "delivery" in c.lower()), date_cols[0] if date_cols else None)
        if target_col:
            like_pattern = f"%{year or ''}%{month_num or ''}%" if year and month_num else f"%{year or month_num}%"
            sql = f"SELECT COUNT(*) FROM {args.table} WHERE {target_col} LIKE ?"
            cur.execute(sql, (like_pattern,))
            count = cur.fetchone()[0]
            print(f"  SQL: {sql}  (param: '{like_pattern}')")
            print(f"  Result: {count} row(s) match {target_col} filter")
            conn.close()
            return

    # --- COUNT ---
    if any(w in question for w in ["how many", "count"]):
        sql = f"SELECT COUNT(*) FROM {args.table}"
        cur.execute(sql)
        print(f"  SQL: {sql}")
        print(f"  Result: {cur.fetchone()[0]} rows")
        conn.close()
        return

    # --- SUM ---
    if any(w in question for w in ["total", "sum", "how much"]):
        num_cols = get_numeric_columns(conn, args.table)
        matched  = find_matching_columns(num_cols, words)
        for col in matched:
            sql = f"SELECT SUM({col}) FROM {args.table}"
            cur.execute(sql)
            print(f"  SQL: {sql}")
            print(f"  Total '{col}': {cur.fetchone()[0]:,.2f}")
        if not matched:
            print("  No matching numeric column. Available numeric columns:")
            for c in num_cols[:10]:
                print(f"    - {c}")
        conn.close()
        return

    # --- AVERAGE ---
    if any(w in question for w in ["average", "avg", "mean"]):
        num_cols = get_numeric_columns(conn, args.table)
        matched  = find_matching_columns(num_cols, words)
        for col in matched:
            sql = f"SELECT AVG({col}) FROM {args.table}"
            cur.execute(sql)
            print(f"  SQL: {sql}")
            print(f"  Average '{col}': {cur.fetchone()[0]:,.2f}")
        conn.close()
        return

    # --- MAX ---
    if any(w in question for w in ["max", "highest", "most", "largest"]):
        num_cols = get_numeric_columns(conn, args.table)
        matched  = find_matching_columns(num_cols, words)
        for col in matched:
            sql = f"SELECT * FROM {args.table} ORDER BY {col} DESC LIMIT 1"
            cur.execute(sql)
            row = cur.fetchone()
            print(f"  SQL: {sql}")
            print(f"  Max '{col}' row: {dict(zip(columns, row))}")
        conn.close()
        return

    # --- MIN ---
    if any(w in question for w in ["min", "lowest", "least", "smallest"]):
        num_cols = get_numeric_columns(conn, args.table)
        matched  = find_matching_columns(num_cols, words)
        for col in matched:
            sql = f"SELECT * FROM {args.table} ORDER BY {col} ASC LIMIT 1"
            cur.execute(sql)
            row = cur.fetchone()
            print(f"  SQL: {sql}")
            print(f"  Min '{col}' row: {dict(zip(columns, row))}")
        conn.close()
        return

    # --- LOOKUP (container/document number etc.) ---
    for word in words:
        if len(word) > 6 and word.replace(".", "").isalnum():
            for col in columns:
                sql = f"SELECT * FROM {args.table} WHERE UPPER({col}) LIKE ?"
                cur.execute(sql, (f"%{word.upper()}%",))
                rows = cur.fetchall()
                if rows:
                    print(f"  SQL: {sql}  (param: '%{word.upper()}%')")
                    print(f"  Found {len(rows)} row(s) in column '{col}':")
                    for r in rows[:5]:
                        print(f"    {dict(zip(columns, r))}")
                    conn.close()
                    return

    # --- SHOW / LIST rows ---
    if any(w in question for w in ["show", "list", "display", "first"]):
        n = next((int(w) for w in words if w.isdigit()), 10)
        sql = f"SELECT * FROM {args.table} LIMIT {n}"
        cur.execute(sql)
        rows = cur.fetchall()
        print(f"  SQL: {sql}")
        for r in rows:
            print(f"    {dict(zip(columns, r))}")
        conn.close()
        return

    # --- SHOW COLUMNS ---
    if any(w in question for w in ["columns", "fields"]):
        print(f"  {len(columns)} columns:")
        for c in columns:
            print(f"    - {c}")
        conn.close()
        return

    # --- DEFAULT ---
    print("  Couldn't match your question to an operation.")
    print("  Try: counts, totals, averages, max/min, a lookup value, or 'show rows'.")
    print(f"\n  Available columns:")
    for c in columns[:20]:
        print(f"    - {c}")
    conn.close()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Structured data engine — SQLite backed")
    sub    = parser.add_subparsers(dest="cmd")

    p_reg = sub.add_parser("register", help="Load an Excel/CSV file into a SQL table")
    p_reg.add_argument("table", help="Table name to create, e.g. 'shipments'")
    p_reg.add_argument("filepath", help="Path to the Excel/CSV file")

    sub.add_parser("list", help="Show all tables")

    p_schema = sub.add_parser("schema", help="Show a table's columns and types")
    p_schema.add_argument("table")

    p_q = sub.add_parser("query", help="Ask a question about a table")
    p_q.add_argument("table")
    p_q.add_argument("question", nargs="+")

    p_sql = sub.add_parser("sql", help="Run raw SQL")
    p_sql.add_argument("query", nargs="+")

    args     = parser.parse_args()
    dispatch = {
        "register": cmd_register,
        "list":     cmd_list,
        "schema":   cmd_schema,
        "query":    cmd_query,
        "sql":      cmd_sql,
    }
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
