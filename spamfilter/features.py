"""Feature engineering: text + simple metadata signals that real spam tends to carry."""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HTML_RE = re.compile(r"<[^>]+>")
MONEY_RE = re.compile(r"[$€£]\s?\d|\d+\s?(usd|birr|etb)\b", re.I)


def clean_text(subject: str, body: str) -> str:
    """Subject is repeated so it carries more weight than a single body word."""
    body = HTML_RE.sub(" ", body or "")
    body = URL_RE.sub(" URLTOKEN ", body)
    subject = URL_RE.sub(" URLTOKEN ", subject or "")
    return f"{subject} {subject} {body}".strip()


class MetaFeatures(BaseEstimator, TransformerMixin):
    """Numeric signals: links, capital letters, exclamation marks, money, HTML, length."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for subject, body in zip(X["subject"].fillna(""), X["body"].fillna("")):
            text = f"{subject} {body}"
            letters = [c for c in text if c.isalpha()]
            caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
            rows.append([
                min(len(URL_RE.findall(text)), 10) / 10,
                caps,
                min(text.count("!"), 10) / 10,
                1.0 if MONEY_RE.search(text) else 0.0,
                1.0 if HTML_RE.search(body) else 0.0,
                min(np.log1p(len(text)) / 10, 1.0),
                1.0 if not subject.strip() else 0.0,
            ])
        return np.array(rows)