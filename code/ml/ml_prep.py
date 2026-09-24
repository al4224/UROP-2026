"""
Build the ML dataset from annotated batches.

Reads the annotated coding files, merges each one with its crawler master for
stratum and weights, splits abstracts into sentences and locates the evidence
sentence, then writes data/ml_dataset.xlsx with two sheets:

  abstracts -- one row per abstract, with the Axis 2 label
  sentences -- one row per sentence, labelled 1 (evidence), 0 (from an N
               abstract) or -1 (unknown: another sentence of a Y abstract)

Run:  python code/ml/ml_prep.py
"""

import sys
from pathlib import Path

import pandas as pd

import ml_sentences as S

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
sys.path.insert(0, str(HERE.parents[0] / "openalex_crawler"))
from openalex_crawler import close_split_hyphens, quality, sdg_keyword, strip_boilerplate  # noqa: E402

# Annotated batches from the crawler: (coding file, master file, split).
# Mark the batch reserved for final evaluation as "test" and never train on it.
BATCHES = [
    ("openalex_batch_20260907_annotated_revised.xlsx",
     "openalex_batch_20260907_master.xlsx", "dev"),
    ("openalex_batch_20260920_annotated_revised.xlsx",
     "openalex_batch_20260920_master.xlsx", "test"),
]
# Hand-collected pilot set: not blind, not from the sampled population, so it is
# kept for sanity checks only and never used for training.
PILOT = "openalex_batch_pilot_annotated_manual.xlsx"
OUT = DATA / "ml_dataset.xlsx"

MASTER_EXTRAS = ["stratum", "population_weight"]


def read(path):
    if not path.exists():
        sys.exit(f"Missing input: {path}")
    return pd.read_excel(path, keep_default_na=False)      # "None" must stay a string


def clean_columns(df, mapping):
    df = df.rename(columns=mapping)
    return df[[c for c in mapping.values() if c in df.columns]]


def load_batch(coding_file, master_file, split):
    df = clean_columns(read(DATA / coding_file), {
        "openalex_id": "openalex_id", "year": "year", "subfield": "subfield",
        "field": "field", "domain": "domain", "abstract": "abstract",
        "contributing_idea": "label", "evidence_sentence": "evidence",
    })
    master_path = DATA / master_file
    if master_path.exists():
        master = read(master_path)[["openalex_id"] + MASTER_EXTRAS]
        df = df.merge(master, on="openalex_id", how="left")
    else:
        print(f"  WARNING: {master_file} not found; no stratum or weights for this batch")
        df[MASTER_EXTRAS] = ""
    return df.assign(source=coding_file, split=split)


def load_pilot(filename):
    df = clean_columns(read(DATA / filename), {
        "OpenAlex_id": "openalex_id", "Year": "year", "Subfield": "subfield",
        "Field": "field", "Domain": "domain", "Abstract": "abstract",
        "Action_contributing_idea": "label", "Contributing_idea_sentences": "evidence",
    })
    df[MASTER_EXTRAS] = ""
    return df.assign(source=filename, split="pilot")


def prepare(df):
    """Normalise text, derive quality, and locate each evidence sentence."""
    stripped = df["abstract"].map(strip_boilerplate)
    df["record_quality"] = [quality(s, s != a) for s, a in zip(stripped, df["abstract"])]
    df["abstract"] = stripped.map(close_split_hyphens)
    df["sdg_keyword"] = df["abstract"].map(sdg_keyword)
    df["label"] = df["label"].str.strip().str.upper()
    df["sentences"] = df["abstract"].map(S.split_sentences)
    df["n_sentences"] = df["sentences"].map(len)
    df["evidence_index"] = [
        S.find_evidence(sents, ev) if lab == "Y" and ev.strip() else None
        for sents, ev, lab in zip(df["sentences"], df["evidence"], df["label"])
    ]
    return df


def to_sentence_rows(df):
    rows = []
    for r in df.itertuples():
        for i, sentence in enumerate(r.sentences):
            if r.label == "N":
                label = 0                    # no sentence in an N abstract qualifies
            elif i == r.evidence_index:
                label = 1
            else:
                label = -1                   # may also qualify; the coder quoted one
            rows.append({"openalex_id": r.openalex_id, "sentence_index": i,
                         "sentence": sentence, "sentence_label": label,
                         "abstract_label": r.label, "split": r.split, "source": r.source})
    return pd.DataFrame(rows)


def main():
    frames = [load_batch(*b) for b in BATCHES] + [load_pilot(PILOT)]
    df = prepare(pd.concat(frames, ignore_index=True))

    unlabelled = df[~df["label"].isin(["Y", "N"])]
    if len(unlabelled):
        print(f"  WARNING: dropping {len(unlabelled)} rows without a Y/N label")
        df = df[df["label"].isin(["Y", "N"])]
    duplicates = df["openalex_id"].duplicated().sum()
    if duplicates:
        print(f"  WARNING: {duplicates} duplicate ids dropped")
        df = df.drop_duplicates("openalex_id")

    missing = df[(df["label"] == "Y") & df["evidence_index"].isna()]
    print(f"\nEvidence located for {int((df['label'] == 'Y').sum()) - len(missing)} "
          f"of {int((df['label'] == 'Y').sum())} Y rows")
    for r in missing.itertuples():
        print(f"  no match: {r.openalex_id} ({r.source})")

    sentences = to_sentence_rows(df)
    abstracts = df.drop(columns=["sentences"])

    DATA.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        abstracts.to_excel(xl, sheet_name="abstracts", index=False)
        sentences.to_excel(xl, sheet_name="sentences", index=False)

    print("\nAbstracts by split and label:")
    print(pd.crosstab(abstracts["split"], abstracts["label"]).to_string())
    print("\nSentences by label (1 evidence / 0 negative / -1 unknown):")
    print(sentences["sentence_label"].value_counts().sort_index().to_string())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
