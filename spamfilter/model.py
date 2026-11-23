"""Model construction, threshold selection and prediction."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import MetaFeatures, clean_text, sender_address, sender_domain

INTERNAL_DOMAIN = "enatbank.com"
ALLOWLIST = {"noreply@enatbank.com"}
BLOCKLIST = set()


def _message_dataframe(subject, body, sender="", channel="email", reply_to="",
                       headers=None, links=None, attachments=None,
                       sender_seen_count=0, redirect_count=0):
    return pd.DataFrame([{
        "subject": subject or "",
        "body": body or "",
        "sender": sender or "",
        "reply_to": reply_to or "",
        "channel": channel or "email",
        "headers": headers or {},
        "links": links or [],
        "attachments": attachments or [],
        "sender_seen_count": sender_seen_count,
        "redirect_count": redirect_count,
    }])


def _list_rule(sender):
    address = sender_address(sender)
    domain = sender_domain(sender)
    allow = set(ALLOWLIST)
    allow.update(x.strip().lower() for x in os.getenv("SPAM_ALLOWLIST", "").split(",") if x.strip())
    block = set(BLOCKLIST)
    block.update(x.strip().lower() for x in os.getenv("SPAM_BLOCKLIST", "").split(",") if x.strip())
    if address in block or (domain and domain in block):
        return "block"
    if address in allow or domain == INTERNAL_DOMAIN:
        return "allow"
    return None


def _combined_text(df: pd.DataFrame):
    return [clean_text(s, b) for s, b in zip(df["subject"].fillna(""), df["body"].fillna(""))]


def build_pipeline() -> Pipeline:
    text = Pipeline([
        ("join", FunctionTransformer(_combined_text)),
        ("tfidf", FeatureUnion([
            # Words catch topics ("lottery", "invoice"); char n-grams catch obfuscation ("fr33", "m0ney").
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])),
    ])
    features = FeatureUnion([("text", text), ("meta", MetaFeatures())])
    # class_weight keeps the model honest when spam/ham are imbalanced.
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))])


def pick_thresholds(y_true, probs, target_precision=0.995):
    """
    Blocking a legitimate email is much worse than letting one spam through, so
    we choose the LOWEST spam-probability cutoff that still reaches the target precision.
      prob >= block      -> SPAM       (auto-move to spam folder)
      review <= prob < block -> SUSPICIOUS (quarantine / warning banner)
      prob <  review     -> HAM        (deliver normally)
    """
    p, r, t = precision_recall_curve(y_true, probs)
    ok = np.where(p[:-1] >= target_precision)[0]
    block = float(t[ok[0]]) if len(ok) else 0.99
    review = float(max(0.5 * block, 0.3))
    return {"block": round(block, 4), "review": round(review, 4)}


class SpamFilter:
    def __init__(self, pipeline, thresholds, meta):
        self.pipeline, self.thresholds, self.meta = pipeline, thresholds, meta
        features = pipeline.named_steps["features"]
        transformers = dict(features.transformer_list)
        meta_features = transformers.get("meta")
        classifier = pipeline.named_steps["clf"]
        if meta_features is not None and hasattr(classifier, "n_features_in_"):
            text = transformers["text"].named_steps["tfidf"]
            text_width = sum(
                len(vectorizer.get_feature_names_out())
                for _, vectorizer in text.transformer_list
            )
            expected_meta_width = classifier.n_features_in_ - text_width
            if expected_meta_width >= 0:
                # Saved models predate the expanded metadata vector; retain their trained shape.
                meta_features.feature_count = expected_meta_width

    def predict(self, subject: str, body: str, sender: str = "", channel: str = "email",
                reply_to: str = "", headers=None, links=None, attachments=None,
                sender_seen_count: int = 0, redirect_count: int = 0) -> dict:
        df = _message_dataframe(subject, body, sender, channel, reply_to, headers,
                                links, attachments, sender_seen_count, redirect_count)
        rule = _list_rule(sender)
        if rule == "allow":
            return {"verdict": "HAM", "spam_probability": 0.0,
                    "model_version": self.meta["version"],
                    "explanation": ["Allowed sender: trusted Enat Bank or allowlist"]}
        if rule == "block":
            return {"verdict": "SPAM", "spam_probability": 1.0,
                    "model_version": self.meta["version"],
                    "explanation": ["Blocked sender: known-bad sender or blocklist"]}
        prob = float(self.pipeline.predict_proba(df)[0][1])
        if prob >= self.thresholds["block"]:
            verdict = "SPAM"
        elif prob >= self.thresholds["review"]:
            verdict = "SUSPICIOUS"
        else:
            verdict = "HAM"
        return {"verdict": verdict, "spam_probability": round(prob, 4),
                "model_version": self.meta["version"], "explanation": self.explain(df)}

    def explain(self, df, n=5):
        """Top words pushing this email toward spam, so admins can understand a decision."""
        feats = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]
        X = feats.transform(df)
        word_vec = feats.transformer_list[0][1].named_steps["tfidf"].transformer_list[0][1]
        n_word = len(word_vec.get_feature_names_out())
        contrib = X[0, :n_word].toarray().ravel() * clf.coef_[0][:n_word]
        names = word_vec.get_feature_names_out()
        top = np.argsort(contrib)[-n:][::-1]
        words = [names[i] for i in top if contrib[i] > 0]
        if hasattr(X, "tocsr"):
            meta = X[0, n_word:].tocsr().toarray().ravel().tolist()
        else:
            meta = np.asarray(X[0, n_word:]).ravel().tolist()
        explanations = []
        if len(meta) > 8 and (meta[7] < 0 or meta[8] < 0):
            explanations.append("Sender fails SPF or DKIM authentication")
        if len(meta) > 9 and meta[9] < 0:
            explanations.append("Sender fails DMARC authentication")
        if len(meta) > 13 and (meta[12] > 0 or meta[13] > 0):
            explanations.append("Display name or Reply-To may be mismatched")
        if len(meta) > 16 and (meta[14] > 0 or meta[15] > 0 or meta[16] > 0):
            explanations.append("Link domain is suspicious or shortened")
        if len(meta) > 18 and meta[18] > 0:
            explanations.append("Risky attachment type detected")
        return explanations + words

    def save(self, directory: str):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, d / "pipeline.joblib")
        (d / "meta.json").write_text(json.dumps({**self.meta, "thresholds": self.thresholds}, indent=2))

    @classmethod
    def load(cls, directory: str):
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        return cls(joblib.load(d / "pipeline.joblib"), meta["thresholds"], meta)