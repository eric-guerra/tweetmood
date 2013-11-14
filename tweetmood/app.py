"""Flask API + dashboard."""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .aggregate import summarize
from .model import SentimentModel, load_dataset

MAX_BATCH = 5000
MAX_TEXT = 1000


def create_app(model: SentimentModel | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

    if model is None:
        path = Path(os.environ.get("TWEETMOOD_MODEL", "model.joblib"))
        model = SentimentModel.load(path) if path.exists() else SentimentModel.train(*load_dataset())
    app.config["MODEL"] = model

    def bad(message: str, status: int = 400):
        return jsonify({"error": message}), status

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        texts = body.get("texts") if "texts" in body else [body.get("text")]
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            return bad("send {text: str} or {texts: [str, ...]}")
        if len(texts) > 100:
            return bad("at most 100 texts per request; use /api/analyze for files")
        texts = [t[:MAX_TEXT] for t in texts]
        return jsonify({"predictions": model.predict(texts)})

    @app.post("/api/analyze")
    def analyze():
        """Upload a CSV with a `text` column (and optional `created_at`) and get dashboard aggregates."""
        upload = request.files.get("file")
        if upload is None:
            return bad("multipart field 'file' is required")
        try:
            reader = csv.DictReader(io.StringIO(upload.read().decode("utf-8-sig")))
        except UnicodeDecodeError:
            return bad("file must be UTF-8 CSV")
        if not reader.fieldnames or "text" not in reader.fieldnames:
            return bad("CSV needs a 'text' column")
        rows = [r for r in reader if (r.get("text") or "").strip()][:MAX_BATCH]
        if not rows:
            return bad("no rows with text")
        preds = model.predict([r["text"][:MAX_TEXT] for r in rows])
        merged = [{"text": r["text"], "created_at": r.get("created_at", ""), "label": p["label"], "polarity": p["polarity"]}
                  for r, p in zip(rows, preds)]
        return jsonify(summarize(merged, bucket=request.args.get("bucket", "hour")))

    @app.get("/api/model")
    def model_card():
        return jsonify({"classes": model.classes, "top_features": model.top_features(12)})

    return app
