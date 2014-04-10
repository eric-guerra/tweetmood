import io

import pytest

from tweetmood.aggregate import summarize
from tweetmood.app import create_app
from tweetmood.model import SentimentModel, evaluate, load_dataset
from tweetmood.preprocess import normalize


@pytest.fixture(scope="module")
def model():
    return SentimentModel.train(*load_dataset())


@pytest.fixture
def client(model):
    return create_app(model).test_client()


def test_normalize_placeholders_and_emoticons():
    toks = normalize("@bob check https://x.co/abc it's sooooo good :) #win 42")
    assert toks[0] == "USER"
    assert "URL" in toks and "EMO_POS" in toks and "NUM" in toks
    assert "ELONG" in toks and "soo" in toks
    assert "HASHTAG" in toks and "win" in toks


def test_negation_scope_ends_at_punctuation():
    toks = normalize("not good at all. great now!")
    assert toks[:4] == ["not", "good_NEG", "at_NEG", "all_NEG"]
    assert "great" in toks and "EXCLAIM" in toks
    assert "don't" in normalize("I don't like it") and "like_NEG" in normalize("I don't like it")


def test_dataset_is_balanced_and_valid():
    texts, labels = load_dataset()
    assert len(texts) == 150
    assert {labels.count(l) for l in set(labels)} == {50}


def test_cross_validated_model_beats_baseline():
    ev = evaluate(*load_dataset(), folds=5)
    assert ev.accuracy > ev.baseline_accuracy + 0.25, ev.report
    assert ev.macro_f1 > 0.6


def test_obvious_examples(model):
    preds = model.predict(["I love this so much :)", "worst experience ever, totally broken :(", "The meeting is at 3pm"])
    assert [p["label"] for p in preds] == ["positive", "negative", "neutral"]
    assert preds[0]["polarity"] > 0 > preds[1]["polarity"]
    assert abs(sum(preds[0]["scores"].values()) - 1) < 1e-3


def test_model_round_trip(model, tmp_path):
    path = tmp_path / "m.joblib"
    model.save(path)
    again = SentimentModel.load(path)
    assert again.predict(["love it"])[0]["label"] == model.predict(["love it"])[0]["label"]


def test_top_features_include_emoticons(model):
    feats = model.top_features(30)
    assert "EMO_POS" in feats["positive"] or "love" in feats["positive"]


def test_predict_api(client):
    r = client.post("/api/predict", json={"texts": ["great job!", "this is awful"]})
    assert r.status_code == 200
    assert [p["label"] for p in r.get_json()["predictions"]] == ["positive", "negative"]
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/api/predict", json={"texts": ["x"] * 101}).status_code == 400


def test_analyze_csv(client):
    csv_text = (
        "text,created_at\n"
        "\"love the new #coffee place :)\",2014-03-01T09:15:00\n"
        "\"worst #coffee ever, cold and bitter\",2014-03-01T09:40:00\n"
        "\"heading to work\",2014-03-01T10:05:00\n"
        ",2014-03-01T10:06:00\n"
    )
    r = client.post("/api/analyze", data={"file": (io.BytesIO(csv_text.encode()), "t.csv")}, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["total"] == 3
    assert [t["bucket"] for t in body["timeline"]] == ["2014-03-01 09:00", "2014-03-01 10:00"]
    assert body["hashtags"][0]["tag"] == "coffee" and body["hashtags"][0]["count"] == 2

    bad = client.post("/api/analyze", data={"file": (io.BytesIO(b"tweet\nhi"), "t.csv")}, content_type="multipart/form-data")
    assert bad.status_code == 400


def test_summarize_handles_twitter_timestamps():
    rows = [{"text": "a", "created_at": "Wed Mar 05 14:02:11 +0000 2014", "label": "positive", "polarity": 0.8},
            {"text": "b", "created_at": "garbage", "label": "negative", "polarity": -0.5}]
    s = summarize(rows, bucket="day")
    assert s["timeline"] == [{"bucket": "2014-03-05", "count": 1, "avg_polarity": 0.8, "positive": 1, "negative": 0, "neutral": 0}]
    assert s["avg_polarity"] == 0.15
