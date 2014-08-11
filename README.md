# TweetMood

A three-class tweet sentiment classifier (positive, neutral, negative) with a **Flask** API and a dashboard. You upload a CSV of tweets and get sentiment over time, per-hashtag polarity, and the most negative and most positive tweets. Built with **scikit-learn** and **NLTK**.

> Portfolio reference implementation of a 2013 NLP project, built using current scikit-learn.

## Approach

Tweet-specific normalization ([tweetmood/preprocess.py](tweetmood/preprocess.py)):
- URLs, @mentions and numbers become placeholder tokens; hashtags keep the word plus a `HASHTAG` marker
- Emoticons and emoji map to `EMO_POS` / `EMO_NEG`
- Elongations are squashed and flagged (`sooooo` → `soo` + `ELONG`); `!` becomes `EXCLAIM`
- **Negation scope**: tokens after *not / never / n't* up to the next punctuation get `_NEG` (`not good` ≠ `good`)

Model: TF-IDF over normalized word uni- and bigrams, **unioned** with character 2–5-grams (robust to slang and typos) → class-balanced logistic regression. Probabilities give a polarity score `P(pos) − P(neg)` that averages cleanly over time windows.

## Results

The bundled dataset is a small, hand-labeled, balanced sample (150 tweets), so expect modest numbers:

```
$ python -m tweetmood.cli evaluate
150 tweets, 5-fold CV
accuracy 0.613 (majority baseline 0.333), macro-F1 0.608
```

Point `--data` at a larger labeled CSV (`label,text`) such as Sentiment140 or SemEval to train a production model.

## Usage

```bash
pip install -r requirements.txt
python -m tweetmood.cli train -o model.joblib
python -m tweetmood.cli predict "not impressed with the update :(" --model model.joblib
TWEETMOOD_MODEL=model.joblib python -m tweetmood.cli serve     # http://localhost:5000
```

API:
- `POST /api/predict` `{"texts": ["..."]}` → label, confidence, polarity, class scores (max 100 per call)
- `POST /api/analyze` multipart CSV (`text`, optional `created_at`) → totals, hourly/daily timeline, hashtags, extremes
- `GET /api/model` → most indicative features per class

## Test

```bash
pytest
```
