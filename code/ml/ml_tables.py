"""
Tables for the writeup.

Two things the analysis needs that the batch files do not hold directly:

  population   per-field, per-stratum population counts from OpenAlex, and the
               field-specific weights they imply. The weight in the batch files
               is pooled across fields, which is correct for an overall rate but
               not for a per-field one.
  labelled     counts of Axis 2 labels by field, stratum and record quality,
               for every labelled batch in ml_dataset.xlsx.

Writes data/ml_tables.xlsx.

Run:  python code/ml/ml_tables.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
sys.path.insert(0, str(HERE.parents[0] / "openalex_crawler"))
from openalex_crawler import FIELD_IDS, count, population_filter  # noqa: E402

FIELD_NAMES = {16: "Chemistry", 26: "Mathematics"}
DATASET = DATA / "ml_dataset.xlsx"
OUT = DATA / "ml_tables.xlsx"


def population_counts():
    """One row per field: works with an abstract, tagged, untagged."""
    rows = []
    for field in FIELD_IDS:
        total = count(population_filter([field]))
        tagged = count(population_filter([field], tagged=True))
        rows.append({"field": FIELD_NAMES.get(field, field),
                     "no_abstract_filter": count(population_filter([field], abstract=False)),
                     "population": total, "tagged": tagged, "untagged": total - tagged,
                     "tagged_share": round(tagged / total, 4)})
    table = pd.DataFrame(rows)
    return pd.concat([table, pd.DataFrame([{
        "field": "Both (pooled)",
        **{c: table[c].sum() for c in ("no_abstract_filter", "population", "tagged", "untagged")},
        "tagged_share": round(table["tagged"].sum() / table["population"].sum(), 4),
    }])], ignore_index=True)


def labelled_counts(abstracts):
    table = (abstracts.groupby(["split", "field", "stratum", "record_quality", "label"])
             .size().unstack("label", fill_value=0))
    table.columns = [f"axis2_{c}" for c in table.columns]
    return table.reset_index()


def field_weights(population, abstracts):
    """
    Weight = population share / sample share, computed within each field.
    The batch files carry the pooled version; these are the per-field ones.
    """
    shares = population.set_index("field")
    rows = []
    for (split, field), group in abstracts.groupby(["split", "field"]):
        if field not in shares.index:
            continue
        for stratum in ("tagged", "untagged"):
            sampled = int((group["stratum"] == stratum).sum())
            if not sampled:
                continue
            population_share = shares.loc[field, stratum] / shares.loc[field, "population"]
            rows.append({"split": split, "field": field, "stratum": stratum,
                         "sampled": sampled, "sample_share": round(sampled / len(group), 4),
                         "population_share": round(population_share, 4),
                         "field_weight": round(population_share / (sampled / len(group)), 4)})
    return pd.DataFrame(rows)


def main():
    if not DATASET.exists():
        sys.exit(f"Missing {DATASET}. Run ml_prep.py first.")
    abstracts = pd.read_excel(DATASET, sheet_name="abstracts", keep_default_na=False)
    abstracts = abstracts[abstracts["stratum"].astype(str).str.strip() != ""]

    print("Querying OpenAlex for per-field population counts...")
    population = population_counts()
    print(population.to_string(index=False))

    labelled = labelled_counts(abstracts)
    print("\nLabelled counts:")
    print(labelled.to_string(index=False))

    weights = field_weights(population, abstracts)
    print("\nField-specific weights:")
    print(weights.to_string(index=False))

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        population.to_excel(xl, sheet_name="population", index=False)
        labelled.to_excel(xl, sheet_name="labelled", index=False)
        weights.to_excel(xl, sheet_name="field_weights", index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
