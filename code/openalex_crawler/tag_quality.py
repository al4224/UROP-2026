"""
Backfill `record_quality` onto an existing coding file, using the same rules as
the crawler. Writes <name>_quality.xlsx next to the input.

  python tag_quality.py openalex_batch_20260912_for_coding.xlsx
"""

import sys
from pathlib import Path

import pandas as pd

from openalex_crawler import quality, strip_boilerplate


def main(path):
    df = pd.read_excel(path, keep_default_na=False)
    raw = df["abstract"]
    stripped = raw.map(strip_boilerplate)
    df["record_quality"] = [quality(s, s != r) for s, r in zip(stripped, raw)]
    df["abstract"] = stripped
    out = Path(path).with_name(Path(path).stem + "_quality.xlsx")
    df.to_excel(out, index=False)
    print(df["record_quality"].value_counts().to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
