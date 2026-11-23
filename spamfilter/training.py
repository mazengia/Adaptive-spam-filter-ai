"""Training logic shared by the command line (train.py) and the web app (Retrain button)."""
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from .model import SpamFilter, build_pipeline, pick_thresholds

# A correction from a person counts this many times more than an ordinary training example.
# Without this, a handful of corrections would be drowned out by thousands of old examples.
FEEDBACK_WEIGHT = 5.0

LABELS = {"spam": 1, "ham": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"subject", "body", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABELS)
    return df.dropna(subset=["label"]).assign(label=lambda d: d["label"].astype(int))


def train_model(data_path, feedback_path, out_dir, target_precision=0.995,
                feedback_weight=FEEDBACK_WEIGHT, version=None, verbose=False) -> dict:
    parts, feedback_rows = [], 0
    if data_path and Path(data_path).exists():
        parts.append(load_csv(data_path).assign(weight=1.0))
    if feedback_path and Path(feedback_path).exists() and Path(feedback_path).stat().st_size > 0:
        fb = load_csv(feedback_path).assign(weight=feedback_weight)
        feedback_rows = len(fb)
        parts.append(fb)
    if not parts:
        raise ValueError("No training data found. Provide data/emails.csv and/or feedback.")

    # keep="last": if a person corrected an email that is also in the base data, the correction wins.
    df = (pd.concat(parts, ignore_index=True)
          .drop_duplicates(["subject", "body"], keep="last").reset_index(drop=True))
    n_spam = int(df["label"].sum())
    n_ham = len(df) - n_spam
    if len(df) < 50 or min(n_spam, n_ham) < 10:
        raise ValueError(f"Need at least 50 emails with 10+ of each class. Found {len(df)} "
                         f"({n_spam} spam, {n_ham} not spam).")
    if verbose:
        print(f"{len(df)} emails, {n_spam / len(df):.1%} spam, {feedback_rows} feedback rows")

    if "date" in df.columns and df["date"].notna().all():
        df = df.sort_values("date")  # train on older mail, test on newer mail
        cut = int(len(df) * 0.8)
        train, test = df.iloc[:cut], df.iloc[cut:]
    else:
        train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train, val = train_test_split(train, test_size=0.2, stratify=train["label"], random_state=1)

    pipe = build_pipeline().fit(train, train["label"], clf__sample_weight=train["weight"].to_numpy())
    thresholds = pick_thresholds(val["label"], pipe.predict_proba(val)[:, 1], target_precision)

    probs = pipe.predict_proba(test)[:, 1]
    pred = (probs >= thresholds["block"]).astype(int)
    auc = float(roc_auc_score(test["label"], probs))
    tn, fp, fn, tp = confusion_matrix(test["label"], pred, labels=[0, 1]).ravel()
    if verbose:
        print(f"Thresholds: {thresholds}   ROC-AUC: {auc:.4f}")
        print(classification_report(test["label"], pred, target_names=["ham", "spam"], zero_division=0))
        print(f"Legitimate emails wrongly blocked: {fp} of {tn + fp}   Spam missed: {fn} of {fn + tp}")

    meta = {"version": version or datetime.now().strftime("%Y%m%d-%H%M%S"),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "train_size": int(len(train)), "test_size": int(len(test)),
            "roc_auc": round(auc, 4), "false_positives": int(fp), "false_negatives": int(fn),
            "legit_in_test": int(tn + fp), "spam_in_test": int(fn + tp),
            "feedback_rows_used": int(feedback_rows)}
    SpamFilter(pipe, thresholds, meta).save(out_dir)
    if verbose:
        print(f"Saved model {meta['version']} to {out_dir}")
    return {**meta, "thresholds": thresholds}