"""CSV reading helpers for datasets that may span older feedback schemas."""
import csv
from pathlib import Path

import pandas as pd


FEEDBACK_COLUMNS = [
    "subject", "body", "label", "date", "sender", "channel", "reply_to",
    "headers", "links", "attachments", "sender_seen_count", "redirect_count",
]


def read_csv_compatible(path) -> pd.DataFrame:
    """Read standard CSV plus feedback rows written before/after metadata was added."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        if not header:
            raise ValueError(f"{path} has an empty CSV header")

        columns = list(header)
        records = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) > len(columns):
                if columns[:4] == FEEDBACK_COLUMNS[:4] and len(row) <= len(FEEDBACK_COLUMNS):
                    columns = FEEDBACK_COLUMNS[:len(row)]
                else:
                    raise ValueError(
                        f"{path} has {len(row)} fields on line {line_number}; "
                        f"expected at most {len(columns)}"
                    )
            records.append(row)

        if len(columns) < len(header):
            columns = list(header)
        records = [row + [""] * (len(columns) - len(row)) for row in records]
        return pd.DataFrame(records, columns=columns)
