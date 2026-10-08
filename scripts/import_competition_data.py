"""Import the competition workbook into the local SQLite database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import DEFAULT_DB_PATH, import_workbook  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Path to the competition .xlsx file")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite output path")
    args = parser.parse_args()
    counts = import_workbook(args.workbook, args.db)
    print(json.dumps({"database": str(args.db), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

