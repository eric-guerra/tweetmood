"""Tweet normalization tuned for sentiment.

* URLs, @mentions and numbers become placeholder tokens (they carry no sentiment but vary wildly).
* Hashtags keep their word (#happy -> happy) and a HASHTAG marker.
* Emoticons and emoji map to EMO_POS / EMO_NEG, which are among the strongest signals in tweets.
* Elongations are squashed (sooooo -> soo) and flagged with ELONG for emphasis.
* Negation scope: tokens after "not", "never", "n't"... up to the next punctuation get a _NEG suffix,
  so "not good" and "good" become different features.
"""
from __future__ import annotations

import re

from nltk.tokenize import TweetTokenizer

_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w{1,15}")
NUMBER_RE = re.compile(r"^\d+([.,]\d+)?$")
ELONG_RE = re.compile(r"(\w)\1{2,}")

POSITIVE_EMOTICONS = {":)", ":-)", ":d", ":-d", ";)", ";-)", "=)", ":p", "<3", "xd", "😀", "😂", "😍", "🙂", "😊", "❤️", "❤", "👍", "🎉", "🔥"}
NEGATIVE_EMOTICONS = {":(", ":-(", ":'(", "d:", ":/", ":-/", ">:(", "😠", "😡", "😢", "😭", "😞", "👎", "💔", "🙁"}
NEGATORS = {"not", "no", "never", "nothing", "nobody", "none", "nowhere", "neither", "nor", "cannot", "without"}
SCOPE_END = re.compile(r"^[.,:;!?]+$")


def normalize(text: str) -> list[str]:
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" USER ", text)
    tokens: list[str] = []
    negating = False
    for raw in _tokenizer.tokenize(text):
        tok = raw.lower()
        if tok in ("url", "user"):
            tokens.append(tok.upper())
            continue
        if tok in POSITIVE_EMOTICONS:
            tokens.append("EMO_POS")
            continue
        if tok in NEGATIVE_EMOTICONS:
            tokens.append("EMO_NEG")
            continue
        if SCOPE_END.match(tok):
            negating = False
            if "!" in tok:
                tokens.append("EXCLAIM")
            continue
        if NUMBER_RE.match(tok):
            tokens.append("NUM")
            continue
        if tok.startswith("#") and len(tok) > 1:
            tokens.append("HASHTAG")
            tok = tok[1:]
        if ELONG_RE.search(tok):
            tokens.append("ELONG")
            tok = ELONG_RE.sub(r"\1\1", tok)
        is_negator = tok in NEGATORS or tok.endswith("n't")
        tokens.append(f"{tok}_NEG" if negating and not is_negator else tok)
        if is_negator:
            negating = True
    return tokens


def analyzer(text: str) -> list[str]:
    """Word unigrams + bigrams over normalized tokens (used as a scikit-learn analyzer)."""
    toks = normalize(text)
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def hashtags(text: str) -> list[str]:
    return [h.lower() for h in re.findall(r"#(\w+)", text)]
