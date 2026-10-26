"""Label the pool papers one at a time: y relevant, n not, s skip, u undo, q quit.

    uv run scripts/label.py
    uv run scripts/label.py --stats

Relevant means it would earn one of the 5 digest slots for its day. Yes: a headline
result, a readable review or tutorial, a method or model likely to matter widely, an
abstract I'd want to keep reading. No, even when the topic is mine: incremental benchmark
gains, narrow technical variants, known methods on niche data, dense specialist jargon.
"""

import argparse
import shutil
import textwrap

from sklearn.metrics import cohen_kappa_score

from recommender.config import ROOT, load_config
from recommender.db import (
    add_label,
    connect,
    export_labels,
    last_labelled,
    next_unlabelled,
    pool_labels,
    queue_paper,
    queue_progress,
    repeat_pairs,
)

try:
    from msvcrt import getwch as read_key # Windows: one keypress, no Enter needed
except ImportError:
    def read_key() -> str:
        return input("> ").strip()[:1]

KEYS = {"y": 1, "n": 0, "s": None}
CLEAR = "\033[2J\033[H"

def show(paper, done: int, total: int, note: str = "") -> None:
    width = min(shutil.get_terminal_size().columns, 110) - 2
    print(CLEAR + f"[{done}/{total}] {note}\n")
    print(textwrap.fill(paper.title, width) + "\n")
    print(f"{paper.categories}    https://arxiv.org/abs/{paper.id}\n")
    print(textwrap.fill(paper.abstract, width) + "\n")
    print("If this were one of today's 5 digest papers, would I be glad it was there?")
    print("y relevant   n not relevant   s skip   u undo   q quit")

def label_loop(conn) -> None:
    position, note = next_unlabelled(conn), ""
    while position is not None:
        done, total = queue_progress(conn)
        show(queue_paper(conn, position), done, total, note)
        key = read_key().lower()
        # getwch reads Ctrl+C as a character instead of raising KeyboardInterrupt.
        if key in ("q", "\x03"):
            return
        if key == "u":
            previous = last_labelled(conn)
            if previous is not None:
                position, note = previous, "(relabelling the previous paper)"
            continue
        if key in KEYS:
            add_label(conn, position, KEYS[key])
            position, note = next_unlabelled(conn), ""
    print("All papers in the queue are labelled.")

def print_stats(conn) -> None:
    done, total = queue_progress(conn)
    print(f"queue: {done}/{total} answered")
    rows = pool_labels(conn)
    for split in ("train", "test"):
        labels = [r[4] for r in rows if r[1] == split]
        known = [x for x in labels if x is not None]
        print(f"{split:<5}: {len(known)}/{len(labels)} labelled, {sum(known)} relevant")
    pairs = repeat_pairs(conn)
    if pairs:
        first, again = zip(*pairs)
        agree = sum(a == b for a, b in pairs) / len(pairs)
        # Kappa is undefined when every answer is the same; report agreement only then.
        kappa = cohen_kappa_score(first, again) if len(set(first + again)) > 1 else float("nan")
        print(f"repeats: {len(pairs)} pairs, {agree:.0%} agreement, Cohen's kappa {kappa:.2f}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="show progress and stop")
    args = parser.parse_args()
    conn = connect(load_config().db_path)
    if args.stats:
        print_stats(conn)
    else:
        label_loop(conn)
        # Keep the committed CSVs in step with the database after every session.
        export_labels(conn, ROOT / "labels")
        print("\nlabels written to labels/*.csv; commit them to keep them safe")
        print_stats(conn)

if __name__ == "__main__":
    main()
