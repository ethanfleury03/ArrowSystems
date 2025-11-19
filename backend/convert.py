import os

INPUT_FILE = "sqlite_dump.sql"      # your original dump
OUTPUT_FILE = "postgres_import.sql" # cleaned output

TABLES = {
    "users",
    "query_history",
    "saved_responses",
    "feedback",
    "audit_logs",
    "alembic_version",
    "machine_models",
    "document_ingestion_metadata",
}

def replace_unistr_in_line(line: str) -> str:
    """Convert SQLite unistr('...') into a normal SQL string literal."""
    out = ""
    i = 0
    while True:
        j = line.find("unistr('", i)
        if j == -1:
            out += line[i:]
            break
        out += line[i:j]
        k = j + len("unistr('")
        s = ""
        # parse until closing '
        while k < len(line):
            ch = line[k]
            if ch == "'":
                # handle escaped ''
                if k + 1 < len(line) and line[k + 1] == "'":
                    s += "'"
                    k += 2
                    continue
                else:
                    k += 1
                    break
            else:
                s += ch
                k += 1
        # interpret \uXXXX etc
        try:
            decoded = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            decoded = s
        # escape single quotes for SQL
        esc = decoded.replace("'", "''")
        out += "'" + esc + "'"
        i = k
    return out


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    # Your dump is UTF-16 (BOM)
    with open(INPUT_FILE, "r", encoding="utf-16") as f:
        lines = f.read().splitlines()

    cleaned = []

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Skip SQLite-specific boilerplate
        if line.startswith("PRAGMA "):
            continue
        if line.startswith("BEGIN TRANSACTION"):
            continue
        if line.startswith("COMMIT"):
            continue
        if line.startswith("CREATE TABLE"):
            continue
        if line.startswith("CREATE INDEX"):
            continue

        # Keep only INSERTs into our tables
        if line.startswith("INSERT INTO "):
            rest = line[len("INSERT INTO "):]
            table_name = rest.split()[0].strip('"')
            if table_name in TABLES:
                if "unistr(" in line:
                    line = replace_unistr_in_line(line)
                cleaned.append(line)

    # Add sequence fix-ups for your SERIAL columns
    cleaned.append("")
    cleaned.append("-- Fix sequences after manual ID inserts")
    cleaned.append(
        "SELECT setval('users_id_seq', "
        "(SELECT COALESCE(MAX(id), 1) FROM users), true);"
    )
    cleaned.append(
        "SELECT setval('query_history_id_seq', "
        "(SELECT COALESCE(MAX(id), 1) FROM query_history), true);"
    )
    cleaned.append(
        "SELECT setval('saved_responses_id_seq', "
        "(SELECT COALESCE(MAX(id), 1) FROM saved_responses), true);"
    )
    cleaned.append(
        "SELECT setval('feedback_id_seq', "
        "(SELECT COALESCE(MAX(id), 1) FROM feedback), true);"
    )
    cleaned.append(
        "SELECT setval('audit_logs_id_seq', "
        "(SELECT COALESCE(MAX(id), 1) FROM audit_logs), true);"
    )
    cleaned.append(
        "SELECT setval('machine_models_id_seq', "
        "(SELECT COALESCE(MAX(id), 1) FROM machine_models), true);"
    )

    # Wrap in a transaction
    output_sql = "BEGIN;\n" + "\n".join(cleaned) + "\nCOMMIT;\n"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_sql)

    print(f"Written cleaned Postgres import script to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
