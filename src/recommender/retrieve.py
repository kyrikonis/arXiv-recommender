import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer

log = logging.getLogger(__name__)

SPECTER_BASE = "allenai/specter2_base"
# Proximity encodes papers; adhoc_query encodes short free-text queries such as the
# interest descriptions. Both produce vectors in the same space.
SPECTER_ADAPTERS = {"proximity": "allenai/specter2", "adhoc_query": "allenai/specter2_adhoc_query"}
MAX_TOKENS = 512
EMBED_DIM = 768


class BM25:
    """Okapi BM25 with document frequencies and average length taken from a fixed corpus."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(stop_words="english")
        tf = self.vectorizer.fit_transform(corpus)
        n = tf.shape[0]
        # Each stored entry in the CSR matrix is one (doc, term) pair, so counting how often
        # a column index appears gives the number of documents containing that term.
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        # The +1 inside the log (as in Lucene) keeps idf positive for very common terms.
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1)
        self.avg_len = tf.sum() / n

    def scores(self, queries: list[str], docs: list[str]) -> np.ndarray:
        """BM25 score of every doc for every query, shape (len(queries), len(docs))."""
        tf = self.vectorizer.transform(docs).tocoo()
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        length_norm = 1 - self.b + self.b * doc_len[tf.row] / self.avg_len
        weight = tf.data * (self.k1 + 1) / (tf.data + self.k1 * length_norm) * self.idf[tf.col]
        doc_term = sp.csr_matrix((weight, (tf.row, tf.col)), shape=tf.shape)
        # Each distinct query term counts once, however often it appears in the query.
        query_term = (self.vectorizer.transform(queries) > 0).astype(np.float64)
        return (query_term @ doc_term.T).toarray()


def zscore_rows(x: np.ndarray) -> np.ndarray:
    """Standardise each row; a constant row becomes all zeros."""
    std = x.std(axis=1, keepdims=True)
    return np.divide(x - x.mean(axis=1, keepdims=True), std, out=np.zeros_like(x), where=std > 0)


def combine(item_scores: np.ndarray, how: str = "max") -> np.ndarray:
    """Collapse (profile items x docs) scores into one score per doc."""
    if how == "max":
        return item_scores.max(axis=0)
    if how == "mean":
        return item_scores.mean(axis=0)
    raise ValueError(f"unknown combine method {how!r}")


def bm25_profile_scores(
    bm25: BM25, queries: list[str], docs: list[str], how: str = "max"
) -> np.ndarray:
    # Raw BM25 grows with query length, so a long seed abstract would dominate a short
    # interest sentence. Standardising each query's scores over the docs puts them on
    # one scale before combining.
    return combine(zscore_rows(bm25.scores(queries, docs)), how)


def dense_profile_scores(profile: np.ndarray, docs: np.ndarray, how: str = "max") -> np.ndarray:
    # Vectors are unit length, so the dot product is the cosine similarity.
    return combine(profile @ docs.T, how)


def paper_text(title: str, abstract: str, sep: str = "[SEP]") -> str:
    return title + sep + abstract


def in_length_order(texts: list[str], encode: Callable[[list[str]], np.ndarray], batch_size: int):
    """Encode texts in batches of similar length, returning rows in the original order.

    Padding every batch to its longest text wastes work, so sorting by length first speeds
    up encoding a lot on abstracts of very different lengths.
    """
    order = np.argsort([len(t) for t in texts], kind="stable")
    parts = [
        encode([texts[i] for i in order[start : start + batch_size]])
        for start in range(0, len(texts), batch_size)
    ]
    out = np.empty((len(texts), parts[0].shape[1]), dtype=parts[0].dtype)
    out[order] = np.concatenate(parts)
    return out


def load_specter(device: str):
    # Imported here so the rest of the module (BM25, scoring) works without loading
    # the adapters library.
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SPECTER_BASE)
    model = AutoAdapterModel.from_pretrained(SPECTER_BASE)
    for name, repo in SPECTER_ADAPTERS.items():
        model.load_adapter(repo, source="hf", load_as=name)
    return tokenizer, model.to(device).eval()


def embed(texts: list[str], tokenizer, model, adapter: str, batch_size: int = 64) -> np.ndarray:
    """Unit-length SPECTER2 embeddings (the [CLS] vector) for each text."""
    model.set_active_adapters(adapter)
    device = model.device

    @torch.inference_mode()
    def encode(batch: list[str]) -> np.ndarray:
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        # Half precision roughly doubles GPU throughput; on CPU it would be slower.
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            cls = model(**tokens).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls.float(), dim=1).cpu().numpy()

    return in_length_order(texts, encode, batch_size)


def save_embeddings(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    np.save(path.with_suffix(".npy"), vectors.astype(np.float32))
    path.with_suffix(".ids").write_text("\n".join(ids))


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.with_suffix(".npy").exists():
        return [], np.empty((0, EMBED_DIM), dtype=np.float32)
    ids = path.with_suffix(".ids").read_text().split("\n")
    vectors = np.load(path.with_suffix(".npy"))
    if len(ids) != len(vectors):
        raise ValueError(f"{path}: {len(ids)} ids but {len(vectors)} vectors")
    return ids, vectors
