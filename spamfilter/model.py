"""Model construction, threshold selection and prediction."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import MetaFeatures, clean_text


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

    def predict(self, subject: str, body: str, sender: str = "") -> dict:
        df = pd.DataFrame({"subject": [subject], "body": [body]})
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
        return [names[i] for i in top if contrib[i] > 0]

    def save(self, directory: str):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, d / "pipeline.joblib")
        (d / "meta.json").write_text(json.dumps({**self.meta, "thresholds": self.thresholds}, indent=2))

    @classmethod
    def load(cls, directory: str):
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        return cls(joblib.load(d / "pipeline.joblib"), meta["thresholds"], meta)