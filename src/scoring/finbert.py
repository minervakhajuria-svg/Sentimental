"""Pass-1 sentiment: FinBERT run locally.

The score is p_pos - p_neg, in [-1, 1]. Using the probability gap rather
than just the argmax label keeps "slightly positive" apart from "very positive",
which matters once scores are averaged across many posts.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Score:
    label: str  # pos | neg | neu
    score: float  # -1..1


class Scorer(ABC):
    """Anything that turns texts into sentiment scores. Tests use a fake one."""

    name: str  # stored as post_scores.model

    @abstractmethod
    def score(self, texts: list[str]) -> list[Score]:
        ...


_LABELS = {"positive": "pos", "negative": "neg", "neutral": "neu"}


def probs_to_scores(probs: list[list[float]], id2label: dict[int, str]) -> list[Score]:
    """Map per-class probabilities to (label, p_pos - p_neg).

    Reads the label order from the model config instead of hardcoding it, so a
    different FinBERT checkpoint with another class order still scores correctly.
    """
    idx = {_LABELS[label.lower()]: i for i, label in id2label.items()}
    out = []
    for p in probs:
        label = max(idx, key=lambda k: p[idx[k]])
        out.append(Score(label, round(float(p[idx["pos"]] - p[idx["neg"]]), 6)))
    return out


class FinBERTScorer(Scorer):
    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 16,
                 max_length: int = 512, device: str | None = None):
        # Imported here: torch is slow to import, and tests never load the model.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.name = "finbert"
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def score(self, texts: list[str]) -> list[Score]:
        out: list[Score] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(**enc).logits, dim=-1).tolist()
            out.extend(probs_to_scores(probs, self.id2label))
        return out
