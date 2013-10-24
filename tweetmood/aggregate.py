"""Turn per-tweet predictions into dashboard series."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from .preprocess import hashtags


def parse_time(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%a %b %d %H:%M:%S %z %Y"):
        try:
            return datetime.strptime(value.strip().replace("Z", "+0000"), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def summarize(rows: list[dict], bucket: str = "hour") -> dict:
    """rows: [{text, created_at?, label, polarity}] -> totals, time series, and per-hashtag sentiment."""
    counts = Counter(r["label"] for r in rows)
    fmt = "%Y-%m-%d %H:00" if bucket == "hour" else "%Y-%m-%d"

    series: dict[str, dict] = defaultdict(lambda: {"n": 0, "polarity": 0.0, "positive": 0, "negative": 0, "neutral": 0})
    tags: dict[str, dict] = defaultdict(lambda: {"n": 0, "polarity": 0.0})
    for r in rows:
        ts = parse_time(r.get("created_at") or "")
        if ts:
            s = series[ts.strftime(fmt)]
            s["n"] += 1
            s["polarity"] += r["polarity"]
            s[r["label"]] += 1
        for tag in set(hashtags(r["text"])):
            tags[tag]["n"] += 1
            tags[tag]["polarity"] += r["polarity"]

    timeline = [{"bucket": k, "count": v["n"], "avg_polarity": round(v["polarity"] / v["n"], 3),
                 "positive": v["positive"], "negative": v["negative"], "neutral": v["neutral"]}
                for k, v in sorted(series.items())]
    top_tags = sorted(({"tag": t, "count": v["n"], "avg_polarity": round(v["polarity"] / v["n"], 3)} for t, v in tags.items()),
                      key=lambda x: (-x["count"], x["tag"]))[:15]
    n = len(rows)
    return {
        "total": n,
        "counts": {k: counts.get(k, 0) for k in ("positive", "neutral", "negative")},
        "avg_polarity": round(sum(r["polarity"] for r in rows) / n, 3) if n else 0.0,
        "timeline": timeline,
        "hashtags": top_tags,
        "most_negative": sorted(rows, key=lambda r: r["polarity"])[:5],
        "most_positive": sorted(rows, key=lambda r: -r["polarity"])[:5],
    }
