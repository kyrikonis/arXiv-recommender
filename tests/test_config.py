# 8 tests convering config. I originally drafted by claude.
from datetime import date
from pathlib import Path

import pytest

from recommender.config import ROOT, load_config, load_profile

VALID = """
categories = ["cs.LG", "quant-ph", "astro-ph.CO"]
digest_size = 5
seed = 42
data_dir = "data"
history_start = 2026-05-01
history_end = 2026-10-23
holidays = [2026-07-03]
"""

PROFILE = """
[ml]
description = "  Machine learning.  "
seeds = ["2501.00663", "1512.03385"]

[quantum]
description = "Quantum computing."
seeds = ["quant-ph/9508027"]
"""

def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path

def test_repo_config_loads():
    cfg = load_config()
    assert cfg.digest_size == 5
    assert "gr-qc" in cfg.categories

def test_valid_config(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.categories == ("cs.LG", "quant-ph", "astro-ph.CO")
    assert cfg.data_dir == ROOT / "data"
    assert cfg.history_start == date(2026, 5, 1)
    assert cfg.holidays == {date(2026, 7, 3)}

@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("history_start = 2026-05-01", 'history_start = "2026-05-01"'),
        ("holidays = [2026-07-03]", 'holidays = ["2026-07-03"]'),
    ],
)
def test_quoted_dates_rejected(tmp_path, old, new):
    with pytest.raises(ValueError, match="unquoted dates"):
        load_config(write(tmp_path, VALID.replace(old, new)))

def test_window_must_be_in_order(tmp_path):
    text = VALID.replace("history_end = 2026-10-23", "history_end = 2026-04-01")
    with pytest.raises(ValueError, match="before history_end"):
        load_config(write(tmp_path, text))

def test_repo_profile_loads():
    areas = load_profile()
    assert {a.name for a in areas} == {"ml", "llm", "quantum", "astro"}
    assert all(a.seeds and a.description for a in areas)

def test_valid_profile(tmp_path):
    areas = load_profile(write(tmp_path, PROFILE))
    assert [a.name for a in areas] == ["ml", "quantum"]
    assert areas[0].description == "Machine learning."
    assert areas[1].seeds == ("quant-ph/9508027",)

@pytest.mark.parametrize("seed", ["2501.00663v2", "arXiv:2501.00663", "2501"])
def test_malformed_seed_rejected(tmp_path, seed):
    text = PROFILE.replace('"1512.03385"', f'"{seed}"')
    with pytest.raises(ValueError, match="malformed"):
        load_profile(write(tmp_path, text))

def test_duplicate_seed_rejected(tmp_path):
    text = PROFILE.replace('"quant-ph/9508027"', '"1512.03385"')
    with pytest.raises(ValueError, match="more than once"):
        load_profile(write(tmp_path, text))

@pytest.mark.parametrize("category", ["cs LG", "cs.LG,", "", "astro-ph..CO"])
def test_malformed_category_rejected(tmp_path, category):
    text = VALID.replace('"quant-ph"', f'"{category}"')
    with pytest.raises(ValueError, match="malformed"):
        load_config(write(tmp_path, text))

def test_duplicate_category_rejected(tmp_path):
    text = VALID.replace('"quant-ph"', '"cs.LG"')
    with pytest.raises(ValueError, match="duplicate"):
        load_config(write(tmp_path, text))

def test_zero_digest_rejected(tmp_path):
    text = VALID.replace("digest_size = 5", "digest_size = 0")
    with pytest.raises(ValueError, match="digest_size"):
        load_config(write(tmp_path, text))
