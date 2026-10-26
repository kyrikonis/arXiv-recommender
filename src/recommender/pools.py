from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

ET = ZoneInfo("America/New_York")
CUTOFF = time(14, 0) # arXiv's daily submission deadline, US Eastern
TEST_BATCHES = 15
TRAIN_BATCHES = 12
POOL_SIZE = 40
TOP_K = 15 # taken from each retriever before filling the pool with random papers
REPEAT_FRACTION = 0.05

@dataclass(frozen=True)
class Batch:
    """Papers submitted in [opens, closes), announced together in one listing."""
    opens: datetime
    closes: datetime

    @property
    def name(self) -> str:
        return self.closes.astimezone(ET).date().isoformat()

def iso(dt: datetime) -> str:
    """UTC timestamp in the same format as Paper.published, so the two compare as text."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def cutoffs(start: date, end: date, holidays: frozenset[date]) -> list[datetime]:
    """Submission deadlines on weekdays from start to end inclusive, skipping holidays."""
    days = (start + timedelta(days=n) for n in range((end - start).days + 1))
    return [
        datetime.combine(d, CUTOFF, ET)
        for d in days
        if d.weekday() < 5 and d not in holidays
    ]

def batches(start: date, end: date, holidays: frozenset[date]) -> list[Batch]:
    """Complete batches inside [start, end) in UTC; batches cut off by the window are dropped."""
    window_start = datetime.combine(start, time(0), UTC)
    window_end = datetime.combine(end, time(0), UTC)
    # Look a few days either side so the batches that straddle the window edges are formed
    # (and then dropped) rather than silently merged into a neighbour.
    cuts = cutoffs(start - timedelta(days=5), end + timedelta(days=5), holidays)
    return [
        Batch(a, b)
        for a, b in zip(cuts, cuts[1:])
        if a >= window_start and b <= window_end
    ]

def split_batches(all_batches: list[Batch], n_test: int, n_train: int):
    """The last n_test batches for testing, and n_train spread evenly over the rest."""
    if n_test + n_train > len(all_batches):
        raise ValueError(f"need {n_test + n_train} batches, have {len(all_batches)}")
    test = all_batches[-n_test:]
    rest = all_batches[:-n_test]
    picks = np.linspace(0, len(rest) - 1, n_train).round().astype(int)
    return [rest[i] for i in picks], test

def build_pool(
    ids: list[str],
    bm25: np.ndarray,
    dense: np.ndarray,
    rng: np.random.Generator,
    size: int = POOL_SIZE,
    top_k: int = TOP_K,
) -> dict[str, str]:
    """Map paper ID to the source that put it in the pool: bm25, dense, both or random."""
    pool = {}
    for i in np.argsort(-bm25, kind="stable")[:top_k]:
        pool[ids[i]] = "bm25"
    for i in np.argsort(-dense, kind="stable")[:top_k]:
        pool[ids[i]] = "both" if ids[i] in pool else "dense"
    # The random part puts papers neither retriever liked in front of the labeller, so
    # the pool isn't only judged on what the two retrievers already agree is relevant.
    rest = [pid for pid in ids if pid not in pool]
    for i in rng.choice(len(rest), min(size - len(pool), len(rest)), replace=False):
        pool[rest[i]] = "random"
    return pool

def build_queue(
    pools: list[list[str]], rng: np.random.Generator, repeat_fraction: float = REPEAT_FRACTION
) -> list[tuple[str, bool]]:
    """Labelling order as (paper_id, is_repeat).

    Pools stay contiguous so an interrupted session leaves only whole pools behind, but
    each is shuffled so a paper's position says nothing about why it was picked. Repeats
    reappear at least one pool later, so they aren't recognised from moments before.
    """
    order = [pool[i] for pool in pools for i in rng.permutation(len(pool))]
    gap = max(len(p) for p in pools)
    eligible = len(order) - 2 * gap
    n_repeats = round(repeat_fraction * len(order))
    if eligible < n_repeats:
        return [(pid, False) for pid in order]
    keyed = [(float(i), pid, False) for i, pid in enumerate(order)]
    for i in rng.choice(eligible, n_repeats, replace=False):
        keyed.append((rng.uniform(i + gap, len(order)), order[i], True))
    return [(pid, repeat) for _, pid, repeat in sorted(keyed)]
