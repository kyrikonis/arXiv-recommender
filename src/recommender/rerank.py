import logging
from collections.abc import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

log = logging.getLogger(__name__)

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_MAX_TOKENS = 384

def fit_logreg(X: np.ndarray, y: np.ndarray, C: float, class_weight: str | None) -> Pipeline:
    # Scaling matters here: the two retrieval scores and the embedding dimensions are on
    # different scales, and L2 regularisation penalises every weight equally.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight=class_weight, max_iter=5000),
    )
    return model.fit(X, y)

class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    hidden: int,
    dropout: float,
    weight_decay: float,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> tuple[MLP, np.ndarray, np.ndarray]:
    """Train one MLP; returns it with the feature mean and std it was trained on."""
    torch.manual_seed(seed)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model = MLP(X.shape[1], hidden, dropout)
    # Relevant papers are a minority; weighting them up keeps the loss from being won by
    # predicting "not relevant" for everything.
    pos_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    shuffle = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(yt), generator=shuffle).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model, mean, std

@torch.inference_mode()
def predict_mlp(models: list[tuple[MLP, np.ndarray, np.ndarray]], X: np.ndarray) -> np.ndarray:
    """Mean predicted probability over an ensemble of independently seeded MLPs."""
    probs = [
        torch.sigmoid(m(torch.tensor((X - mean) / std, dtype=torch.float32))).numpy()
        for m, mean, std in models
    ]
    return np.mean(probs, axis=0)

def profile_query(descriptions: list[str]) -> str:
    """The cross-encoder's fixed query: my interest areas as one statement."""
    return " ".join(descriptions)

def _tokenize(tokenizer, query: str, texts: list[str], device: str):
    # truncation="only_second" cuts the paper, never the query, when the pair is too long.
    return tokenizer(
        [query] * len(texts),
        texts,
        padding=True,
        truncation="only_second",
        max_length=CROSS_MAX_TOKENS,
        return_tensors="pt",
    ).to(device)

def load_cross(name_or_path: str, device: str):
    """The pretrained cross-encoder by name, or a fine-tuned one saved by train.py."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path).to(device)
    return model, tokenizer

@torch.inference_mode()
def predict_cross(model, tokenizer, query: str, texts: list[str], device: str, batch_size: int = 32):
    model.eval()
    scores = []
    for start in range(0, len(texts), batch_size):
        tokens = _tokenize(tokenizer, query, texts[start : start + batch_size], device)
        scores.append(model(**tokens).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(scores)

def fit_cross(
    query: str,
    texts: list[str],
    y: np.ndarray,
    seed: int,
    lr: float,
    epochs: int,
    device: str,
    after_epoch: Callable[[int, object, object], None] | None = None,
    batch_size: int = 16,
):
    """Fine-tune the pretrained cross-encoder on (query, paper) -> relevant.

    after_epoch(epoch, model, tokenizer) is called after each epoch, so cross-validation
    can score every epoch count from one run instead of retraining for each.
    """
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    model, tokenizer = load_cross(CROSS_ENCODER, device)
    yt = torch.tensor(y, dtype=torch.float32)
    pos_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = -(-len(texts) // batch_size)
    # Warm up over the first 10% of steps: large updates at the start can wreck the
    # pretrained weights before the new task has any signal.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * steps_per_epoch * epochs), steps_per_epoch * epochs
    )
    shuffle = torch.Generator().manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(texts), generator=shuffle).split(batch_size):
            tokens = _tokenize(tokenizer, query, [texts[i] for i in idx], device)
            # bfloat16 autocast speeds up the GPU without the loss scaling float16 needs.
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**tokens).logits.squeeze(-1)
            loss = loss_fn(logits.float(), yt[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if after_epoch:
            after_epoch(epoch, model, tokenizer)
    return model, tokenizer

def load_mlp(path) -> list[tuple[MLP, np.ndarray, np.ndarray]]:
    """The MLP ensemble saved by train.py, ready for predict_mlp."""
    saved = torch.load(path, weights_only=False)
    config = saved["config"]
    models = []
    for state, mean, std in zip(saved["states"], saved["means"], saved["stds"]):
        model = MLP(len(mean), config["hidden"], config["dropout"])
        model.load_state_dict(state)
        model.eval()
        models.append((model, mean, std))
    return models

def predict_cross_ensemble(models, query: str, texts: list[str], device: str) -> np.ndarray:
    """Mean logit over the fine-tuned seeds."""
    return np.mean([predict_cross(m, tok, query, texts, device) for m, tok in models], axis=0)
