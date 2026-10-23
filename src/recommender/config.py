import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Archive, optionally followed by a subject class: "gr-qc", "cs.LG", "astro-ph.CO".
CATEGORY_RE = re.compile(r"^[a-z-]+(\.[A-Za-z-]+)?$")


@dataclass(frozen=True)
class Config:
    categories: tuple[str, ...]
    digest_size: int
    seed: int
    data_dir: Path
    interests: str


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

    # Relative paths are resolved against the repo root so scripts work from any cwd.
    data_dir = Path(raw["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    return Config(
        categories=categories,
        digest_size=digest_size,
        seed=raw["seed"],
        data_dir=data_dir,
        interests=raw["interests"].strip(),
    )
