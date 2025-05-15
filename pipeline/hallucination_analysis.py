"""
Compare two caption CSVs and write out rows whose transcriptions changed.

Input files  (edit the paths below if needed)
  processed_output/output_transcription_translated_final.csv
  processed_output/output_transcription_translated_manually_cleaned_final.csv

Output file
  processed_output/transcription_changes.csv
"""

import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. Paths – edit if your folder structure is different
orig_path  = Path("processed_output/output_transcription_translated_final.csv")
fixed_path = Path("processed_output/output_transcription_translated_manually_cleaned_final.csv")
out_path   = Path("processed_output/transcription_changes.csv")

# --------------------------------------------------------------------------- #
# 2. Load the two CSVs
orig  = pd.read_csv(orig_path,  dtype=str)   # read everything as str to avoid NaN vs nan quirks
fixed = pd.read_csv(fixed_path, dtype=str)

# --------------------------------------------------------------------------- #
# 3. Merge on the unique key (“id”) so we can compare columns side-by-side
df = (
    orig.rename(columns={"transcription": "orig_transcription",
                         "translation":   "orig_translation"})
        .merge(
            fixed.rename(columns={"transcription": "fixed_transcription",
                                  "translation":   "fixed_translation"}),
            on="id",
            suffixes=("_origDROP", "_fixedDROP"),   # guard against unexpected dups
            how="inner"
        )
)

# --------------------------------------------------------------------------- #
# 4. Flag rows where the transcriptions are different (whitespace-insensitive)
def norm(x: pd.Series) -> pd.Series:
    return x.fillna("").str.strip()

mask = norm(df["orig_transcription"]) != norm(df["fixed_transcription"])
changes = df.loc[mask]

# --------------------------------------------------------------------------- #
# 5. Build the output table
#    • Keep every column except the now-redundant *_origDROP / *_fixedDROP
#    • Use orig_* columns where possible; fall back to fixed_* if an extra col exists
base_cols   = [c for c in orig.columns if c not in ("transcription", "translation")]
extra_cols  = [c for c in fixed.columns if c not in ("transcription", "translation") and c not in base_cols]
ordered_cols = base_cols + extra_cols

out = pd.DataFrame()

for col in ordered_cols:
    if f"{col}_origDROP" in changes:
        out[col] = changes[f"{col}_origDROP"]
    elif col in changes:                 # if merge didn’t duplicate this column
        out[col] = changes[col]
    elif f"{col}_fixedDROP" in changes:  # column exists only in the fixed file (e.g., Goodness)
        out[col] = changes[f"{col}_fixedDROP"]

# Add the four text columns
out["orig_transcription"]  = changes["orig_transcription"]
out["orig_translation"]    = changes["orig_translation"]
out["fixed_transcription"] = changes["fixed_transcription"]
out["fixed_translation"]   = changes["fixed_translation"]

# --------------------------------------------------------------------------- #
# 6. Save
out.to_csv(out_path, index=False)
print(f"Wrote {len(out)} changed rows to {out_path.resolve()}")
