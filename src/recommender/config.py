import re
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Archive, optionally followed by a subject class: "gr-qc", "cs.LG", "astro-ph.CO".
CATEGORY_RE = re.compile(r"^[a-z-]+(\.[A-Za-z-]+)?$")
# New-style "2501.00663" or old-style "quant-ph/9508027", without a version suffix.
ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})$")

@dataclass(frozen=True)
class Config:
    categories: tuple[str, ...]
    digest_size: int
    seed: int
    data_dir: Path
    history_start: date
    history_end: date
    holidays: frozenset[date]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "papers.db"

@dataclass(frozen=True)
class Area:
    name: str
    description: str
    seeds: tuple[str, ...]

def load_config(path: Path = ROOT / "config.toml") -> Config:
    with path.open("rb") as f:
        raw = tomllib.load(f)

    categories = tuple(raw["categories"])
    if not categories:
        raise ValueError("config: categories is empty")
    bad = [c for c in categories if not CATEGORY_RE.match(c)]
    if bad:
        raise ValueError(f"config: malformed categories {bad}")
    if len(set(categories)) != len(categories):
        raise ValueError("config: duplicate categories")

    digest_size = raw["digest_size"]
    if digest_size < 1:
        raise ValueError("config: digest_size must be at least 1")

    # TOML has a native date type; a quoted string here would be a mistake.
    dates = [raw["history_start"], raw["history_end"], *raw["holidays"]]
    if not all(isinstance(d, date) for d in dates):
        raise ValueError("config: history_start, history_end and holidays must be unquoted dates")
    if raw["history_start"] >= raw["history_end"]:
        raise ValueError("config: history_start must be before history_end")

    # Relative paths are resolved against the repo root so scripts work from any cwd.
    data_dir = Path(raw["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    return Config(
        categories=categories,
        digest_size=digest_size,
        seed=raw["seed"],
        data_dir=data_dir,
        history_start=raw["history_start"],
        history_end=raw["history_end"],
        holidays=frozenset(raw["holidays"]),
    )

def load_profile(path: Path = ROOT / "profile.toml") -> list[Area]:
    with path.open("rb") as f:
        raw = tomllib.load(f)

    areas = [Area(name, a["description"].strip(), tuple(a["seeds"])) for name, a in raw.items()]
    seeds = [s for a in areas for s in a.seeds]
    bad = [s for s in seeds if not ARXIV_ID_RE.match(s)]
    if bad:
        raise ValueError(f"profile: malformed or versioned seed IDs {bad}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("profile: a seed appears more than once")
    return areas
