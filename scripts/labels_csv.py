"""Write the pools, queue and labels to labels/*.csv, or load them into a fresh database.

    uv run scripts/labels_csv.py export
    uv run scripts/labels_csv.py import    # after fetch_history.py on a fresh clone

label.py exports automatically when it exits, so export is rarely needed by hand.
"""

import argparse

from recommender.config import ROOT, load_config
from recommender.db import connect, export_labels, import_labels, queue_progress

LABELS_DIR = ROOT / "labels"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["export", "import"])
    args = parser.parse_args()
    conn = connect(load_config().db_path)
    if args.action == "export":
        export_labels(conn, LABELS_DIR)
    else:
        import_labels(conn, LABELS_DIR)
    done, total = queue_progress(conn)
    print(f"{args.action}ed {LABELS_DIR}: {done}/{total} queue positions answered")

if __name__ == "__main__":
    main()
