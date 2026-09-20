"""
Rebuild processed_ids.txt from every batch file in the data folder.

The crawler skips ids listed there, so if the file is missing, stale or was
left behind by a folder move, an earlier batch can be served again. This
collects the ids out of every spreadsheet that has an openalex_id column and
merges them into the list, keeping whatever is already there.

Run:  python code/openalex_crawler/rebuild_processed_ids.py
"""

import pandas as pd

from openalex_crawler import PROCESSED_IDS_FILE

DATA = PROCESSED_IDS_FILE.parent


def ids_in(path):
    for sheet in pd.read_excel(path, sheet_name=None, keep_default_na=False).values():
        column = next((c for c in sheet.columns
                       if c.strip().lower() in ("openalex_id", "id")), None)
        if column is not None:
            yield from (str(v).strip() for v in sheet[column] if str(v).strip())


def main():
    known = set()
    if PROCESSED_IDS_FILE.exists():
        known = set(PROCESSED_IDS_FILE.read_text(encoding="utf-8").split())
    print(f"{len(known)} ids already listed")

    found = {}
    for path in sorted(DATA.glob("*.xlsx")):
        if path.name.startswith(("ml_", "~$")):
            continue
        ids = set(ids_in(path))
        if ids:
            found[path.name] = ids
            print(f"  {path.name}: {len(ids)} ids, {len(ids - known)} new")

    everything = known.union(*found.values()) if found else known
    PROCESSED_IDS_FILE.write_text("\n".join(sorted(everything)) + "\n", encoding="utf-8")
    print(f"\nWrote {PROCESSED_IDS_FILE} with {len(everything)} ids "
          f"({len(everything) - len(known)} added)")


if __name__ == "__main__":
    main()
