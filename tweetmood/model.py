"""Training, evaluation and persistence for the sentiment classifier."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from . import LABELS
from .preprocess import analyzer

DATA = Path(__file__).resolve().parent.parent / "data" / "tweets.csv"


def load_dataset(path: Path = DATA) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = row["label"].strip().lower()
            if label not in LABELS:
                raise ValueError(f"unknown label {label!r} in {path}")
            texts.append(row["text"])
            labels.append(label)
    return texts, labels


def build_pipeline(C: float = 4.0) -> Pipeline:
    """Normalized word n-grams plus raw character n-grams (robust to typos and slang)."""
    features = FeatureUnion([
        ("words", TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, min_df=1)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True, min_df=1)),
    ])
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class Evaluation:
    macro_f1: float
    accuracy: float
    baseline_accuracy: float
    report: str
    confusion: list[list[int]]


def evaluate(texts: list[str], labels: list[str], folds: int = 5, seed: int = 13) -> Evaluation:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = cross_val_predict(build_pipeline(), texts, labels, cv=cv)
    y = np.array(labels)
    majority = max(set(labels), key=labels.count)
    return Evaluation(
        macro_f1=float(f1_score(y, preds, average="macro")),
        accuracy=float((preds == y).mean()),
        baseline_accuracy=labels.count(majority) / len(labels),
        report=classification_report(y, preds, labels=list(LABELS), zero_division=0),
        confusion=confusion_matrix(y, preds, labels=list(LABELS)).tolist(),
    )


class SentimentModel:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> "SentimentModel":
        return cls(build_pipeline().fit(texts, labels))

    @classmethod
    def load(cls, path: str | Path) -> "SentimentModel":
        return cls(joblib.load(path))

    def save(self, path: str | Path) -> None:
        joblib.dump(self.pipeline, path)

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    def predict(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        probs = self.pipeline.predict_proba(texts)
        out = []
        for text, p in zip(texts, probs):
            scores = {c: round(float(v), 4) for c, v in zip(self.classes, p)}
            label = max(scores, key=scores.get)
            # Polarity in [-1, 1]: useful for averaging over time.
            polarity = scores.get("positive", 0.0) - scores.get("negative", 0.0)
            out.append({"text": text, "label": label, "confidence": scores[label], "polarity": round(polarity, 4), "scores": scores})
        return out

    def top_features(self, n: int = 15) -> dict[str, list[str]]:
        """Most indicative word features per class (for the model card on the dashboard)."""
        words = self.pipeline.named_steps["features"].transformer_list[0][1]
        names = np.array(words.get_feature_names_out())
        coefs = self.pipeline.named_steps["clf"].coef_[:, : len(names)]
        return {cls: names[np.argsort(row)[::-1][:n]].tolist() for cls, row in zip(self.classes, coefs)}
