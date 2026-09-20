"""
Where does the baseline go wrong?

Scores every dev abstract with a model that never saw it (out-of-fold), then
breaks the errors down and writes them out for reading. Two questions:
are the mistakes genuinely hard cases or label problems, and would a
confidence threshold let a model handle the easy rows on its own?

Run:  python code/ml/ml_errors.py
"""

import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import ml_baseline
from ml_baseline import DATA, FOLDS, SEED, load

KIND = sys.argv[1] if len(sys.argv) > 1 else "tfidf"     # tfidf | embed
OUT = DATA / f"ml_errors_{KIND}.xlsx"
UNCERTAIN = (0.35, 0.65)     # band to send for human review under triage


def make_scorer(dev, sentences):
    """The chosen model, as a function of (train rows, test rows) -> scores."""
    if KIND == "tfidf":
        return lambda train, test: ml_baseline.sentence_scores(train, test, sentences)
    import ml_embed
    data = ml_embed.prepare_data(dev, sentences)
    return lambda train, test: ml_embed.abstract_scores(train, test, data)


def out_of_fold(dev, sentences):
    """Score each row with a model trained without it."""
    scorer = make_scorer(dev, sentences)
    scores = pd.Series(index=dev.index, dtype=float)
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in folds.split(dev, dev["y"]):
        scores.iloc[test_idx] = scorer(dev.iloc[train_idx], dev.iloc[test_idx])
    return scores


def breakdown(dev, column):
    table = dev.groupby(column).agg(rows=("y", "size"), errors=("wrong", "sum"))
    table["error_rate"] = (table["errors"] / table["rows"]).round(3)
    return table.sort_values("errors", ascending=False)


def main():
    dev, sentences = load()
    dev["y"] = (dev["label"] == "Y").astype(int)
    dev["score"] = out_of_fold(dev, sentences)
    dev["predicted"] = (dev["score"] >= 0.5).astype(int)
    dev["wrong"] = dev["predicted"] != dev["y"]
    dev["error_type"] = [
        "" if not w else ("false_Y" if p else "false_N")
        for w, p in zip(dev["wrong"], dev["predicted"])]

    print(f"\nmodel: {KIND}")
    print(f"{len(dev)} dev rows, {int(dev['wrong'].sum())} wrong "
          f"({1 - dev['wrong'].mean():.1%} accurate)")
    print(pd.crosstab(dev["label"], dev["predicted"].map({0: "pred_N", 1: "pred_Y"})).to_string())

    for column in ("field", "stratum", "record_quality"):
        print(f"\nBy {column}:")
        print(breakdown(dev, column).to_string())

    low, high = UNCERTAIN
    uncertain = dev["score"].between(low, high)
    print(f"\nConfidence band {low}-{high}: {uncertain.sum()} rows "
          f"({uncertain.mean():.0%}), accuracy inside {1 - dev.loc[uncertain, 'wrong'].mean():.1%}, "
          f"outside {1 - dev.loc[~uncertain, 'wrong'].mean():.1%}")

    columns = ["openalex_id", "label", "predicted", "score", "error_type", "field",
               "subfield", "stratum", "record_quality", "evidence", "abstract"]
    errors = dev[dev["wrong"]].sort_values("score", key=lambda s: (s - 0.5).abs(),
                                           ascending=False)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        errors[columns].to_excel(xl, sheet_name="errors", index=False)
        dev[columns].to_excel(xl, sheet_name="all_scored", index=False)
    print(f"\nWrote {OUT} ({len(errors)} errors, most confident first)")


if __name__ == "__main__":
    main()
