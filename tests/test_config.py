# 8 tests convering config. I originally drafted by claude.
from pathlib import Path
import pytest

from recommender.config import ROOT, load_config

VALID = """
categories = ["cs.LG", "quant-ph", "astro-ph.CO"]
digest_size = 5
seed = 42
data_dir = "data"
interests = "  machine learning  "
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
    assert cfg.interests == "machine learning"

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