"""File-based storage for user feedback and model versions (no web framework needed)."""
import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .csv_utils import FEEDBACK_COLUMNS, read_csv_compatible


class Store:
    def __init__(self, feedback_file, models_root):
        self.feedback_file = Path(feedback_file)
        self.models_root = Path(models_root)
        self._lock = threading.Lock()

    # ---------- feedback ----------
    def add_feedback(self, subject: str, body: str, label: str, sender: str = "",
                     channel: str = "email", reply_to: str = "", headers=None,
                     links=None, attachments=None, sender_seen_count: int = 0,
                     redirect_count: int = 0):
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            new = not self.feedback_file.exists() or self.feedback_file.stat().st_size == 0
            if not new:
                # Normalize mixed legacy/new rows before appending, and update the header.
                existing = read_csv_compatible(self.feedback_file).reindex(
                    columns=FEEDBACK_COLUMNS, fill_value=""
                )
                existing.to_csv(self.feedback_file, index=False, encoding="utf-8")
            with self.feedback_file.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(FEEDBACK_COLUMNS)
                w.writerow([subject, body, label,
                            datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            sender, channel, reply_to, json.dumps(headers or {}),
                            json.dumps(links or []), json.dumps(attachments or []),
                            sender_seen_count, redirect_count])

    def feedback_stats(self, used: int = 0, recent_n: int = 20) -> dict:
        with self._lock:
            if not self.feedback_file.exists() or self.feedback_file.stat().st_size == 0:
                return {"total": 0, "spam": 0, "ham": 0, "pending": 0, "recent": []}
            df = read_csv_compatible(self.feedback_file).fillna("")
        recent = [{"subject": str(r.subject)[:90], "label": str(r.label), "date": str(r.date)}
                  for r in df.tail(recent_n).iloc[::-1].itertuples()]
        total = len(df)
        return {"total": total, "spam": int((df["label"] == "spam").sum()),
                "ham": int((df["label"] == "ham").sum()),
                "pending": max(total - used, 0), "recent": recent}

    # ---------- model versions ----------
    def list_models(self, active_version: str) -> list:
        out = []
        if self.models_root.exists():
            for d in self.models_root.iterdir():
                mf = d / "meta.json"
                if d.is_dir() and mf.exists():
                    m = json.loads(mf.read_text())
                    out.append({"name": d.name, "version": m.get("version", ""),
                                "trained_at": m.get("trained_at", ""),
                                "roc_auc": m.get("roc_auc"),
                                "false_positives": m.get("false_positives"),
                                "false_negatives": m.get("false_negatives"),
                                "feedback_rows_used": m.get("feedback_rows_used", 0),
                                "active": m.get("version") == active_version})
        return sorted(out, key=lambda m: m["version"], reverse=True)

    def resolve(self, name: str):
        d = self.models_root / Path(name).name  # .name blocks path traversal like ../../x
        return d if (d / "meta.json").exists() else None

    def active_dir(self, default) -> Path:
        """The model that was last activated or trained survives a restart."""
        p = self.models_root / "active.txt"
        if p.exists():
            d = self.resolve(p.read_text().strip())
            if d:
                return d
        return Path(default)

    def set_active(self, name: str):
        self.models_root.mkdir(parents=True, exist_ok=True)
        (self.models_root / "active.txt").write_text(Path(name).name)