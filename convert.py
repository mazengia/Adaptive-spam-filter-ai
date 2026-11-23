import pandas as pd
from pathlib import Path

# Project data directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

input_file = DATA_DIR / "SMSSpamCollection"
output_file = DATA_DIR / "emails.csv"

# Check whether dataset exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Dataset not found: {input_file}\n"
        "Please put SMSSpamCollection inside the data directory."
    )

# Read SMS spam dataset
df = pd.read_csv(
    input_file,
    sep="\t",
    names=["label", "body"]
)

# Add an empty subject column
df["subject"] = ""

# Export as CSV
df[["subject", "body", "label"]].to_csv(
    output_file,
    index=False
)

print(f"Conversion successful! Saved to: {output_file}")
print(df.head())