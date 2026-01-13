from __future__ import annotations

from pathlib import Path

# Extensions to count (edit as needed)
CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".go", ".rs", ".php", ".rb",
    ".kt", ".swift", ".sql", ".sh", ".ps1",
    ".html", ".css", ".scss", ".json", ".yml", ".yaml", ".toml", ".md",
}

# Folders to skip (edit as needed)
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__",
    "node_modules", "dist", "build", ".next", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
}

def is_in_skipped_dir(path: Path) -> bool:
    parts = set(path.parts)
    return any(d in parts for d in SKIP_DIRS)

def count_lines(p: Path) -> int:
    try:
        # binary-safe-ish: ignores bad chars
        return sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
    except (OSError, UnicodeError):
        return 0

def main() -> None:
    root = Path(__file__).resolve().parent
    total_lines = 0
    total_files = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_in_skipped_dir(p):
            continue
        if p.suffix.lower() not in CODE_EXTS:
            continue

        total_files += 1
        total_lines += count_lines(p)

    print(f"Root: {root}")
    print(f"Counted files: {total_files}")
    print(f"Total lines: {total_lines}")

if __name__ == "__main__":
    main()
