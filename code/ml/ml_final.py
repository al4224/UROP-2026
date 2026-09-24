"""
The frozen pipeline, evaluated once against the sealed test batch.

Configuration below was fixed on the dev rows before any test row was scored.
Do not tune it against the output of this script: the whole value of the test
batch is that nothing was chosen using it.

  1. embeddings + logistic regression, trained on every dev row
  2. abstracts scoring inside the uncertain band go to the local LLM
  3. accuracy, per-stratum error, and a population estimate corrected for the
     measured error rate

Writes data/ml_final_results.xlsx.

Run:  python code/ml/ml_final.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import ml_llm
from ml_baseline import DATA, score
from ml_embed import MODEL_NAME, classifier, prepare_data

# ---- frozen configuration --------------------------------------------------
BAND = (0.40, 0.60)          # send to the LLM; chosen on dev, mid-plateau
THRESHOLD = 0.5
OUT = DATA / "ml_final_results.xlsx"
BOOTSTRAP = 2000


def interval(correct, resamples=BOOTSTRAP, seed=0):
    """Bootstrap 95% interval for a proportion."""
    rng = np.random.default_rng(seed)
    values = np.asarray(correct, dtype=float)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def breakdown(frame, column):
    table = frame.groupby(column).agg(rows=("wrong", "size"), errors=("wrong", "sum"))
    table["error_rate"] = (table["errors"] / table["rows"]).round(3)
    return table


def corrected_rate(observed, sensitivity, specificity):
    """
    Rogan-Gladen: recover the true rate from a measured one when the classifier
    is known to be imperfect. Undefined if the two rates do not beat chance.
    """
    denominator = sensitivity + specificity - 1
    if denominator <= 0:
        return float("nan")
    return min(max((observed + specificity - 1) / denominator, 0.0), 1.0)


def weighted(values, weights):
    return float(np.average(values, weights=weights))


def main():
    abstracts = pd.read_excel(DATA / "ml_dataset.xlsx", sheet_name="abstracts",
                              keep_default_na=False)
    sentences = pd.read_excel(DATA / "ml_dataset.xlsx", sheet_name="sentences",
                              keep_default_na=False)
    abstracts["y"] = (abstracts["label"] == "Y").astype(int)

    empty = abstracts["abstract"].str.strip() == ""
    if empty.any():
        print(f"Excluding {int(empty.sum())} rows with no abstract text: "
              f"{', '.join(abstracts.loc[empty, 'openalex_id'])}")
        abstracts = abstracts[~empty]

    dev = abstracts[abstracts["split"] == "dev"].reset_index(drop=True)
    test = abstracts[abstracts["split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise SystemExit("No rows with split 'test'. Add the batch to BATCHES in "
                         "ml_prep.py and rerun it.")
    missing_weights = test["population_weight"].astype(str).str.strip() == ""
    if missing_weights.any():
        raise SystemExit(f"{int(missing_weights.sum())} test rows have no "
                         "population_weight; the master file did not merge.")
    print(f"train on {len(dev)} dev rows, evaluate on {len(test)} test rows\n")

    both = pd.concat([dev, test], ignore_index=True)
    data = prepare_data(both, sentences[sentences["openalex_id"].isin(both["openalex_id"])])
    vectors = data["abstract_x"]
    model = classifier().fit(vectors[:len(dev)], dev["y"])
    test["score"] = model.predict_proba(vectors[len(dev):])[:, 1]
    test["embed_pred"] = (test["score"] >= THRESHOLD).astype(int)

    low, high = BAND
    uncertain = test["score"].between(low, high)
    print(f"{int(uncertain.sum())} of {len(test)} test rows fall in the "
          f"{low}-{high} band and go to {ml_llm.MODEL}")
    ml_llm.CACHE = DATA / f"ml_llm_cache_{ml_llm.MODEL.replace(':', '_')}_test.json"
    answers = ml_llm.judge(test[uncertain])
    test["llm_pred"] = [int(answers.get(i, {}).get("label") == "Y") for i in test["openalex_id"]]
    test["predicted"] = np.where(uncertain, test["llm_pred"], test["embed_pred"])
    test["wrong"] = test["predicted"] != test["y"]

    metrics = score(test["y"].values, test["predicted"].values)
    metrics["kappa"] = cohen_kappa_score(test["y"], test["predicted"])
    lower, upper = interval(~test["wrong"])
    print("\n=== held-out test results ===")
    print(pd.Series(metrics).round(3).to_string())
    print(f"accuracy 95% CI: [{lower:.3f}, {upper:.3f}]")
    print(f"errors: {int(test['wrong'].sum())} "
          f"({int(((test.predicted == 1) & (test.y == 0)).sum())} false Y, "
          f"{int(((test.predicted == 0) & (test.y == 1)).sum())} false N)")

    for column in ("stratum", "field", "record_quality"):
        print(f"\nBy {column}:")
        print(breakdown(test, column).to_string())

    # --- population estimate ------------------------------------------------
    weights = pd.to_numeric(test["population_weight"])
    truth = weighted(test["y"], weights)
    observed = weighted(test["predicted"], weights)
    sensitivity = weighted((test["predicted"] == 1)[test["y"] == 1], weights[test["y"] == 1])
    specificity = weighted((test["predicted"] == 0)[test["y"] == 0], weights[test["y"] == 0])
    estimate = corrected_rate(observed, sensitivity, specificity)

    print("\n=== weighted population estimate ===")
    print(f"sensitivity {sensitivity:.3f}, specificity {specificity:.3f}")
    print(f"rate from labels     {truth:.3f}   <- the annotated truth for this batch")
    print(f"rate from model      {observed:.3f}")
    print(f"error-corrected      {estimate:.3f}   <- apply this correction at scale")

    summary = pd.DataFrame([
        {"quantity": k, "value": v} for k, v in {
            "embedding_model": MODEL_NAME, "llm_model": ml_llm.MODEL,
            "band_low": low, "band_high": high, "train_rows": len(dev),
            "test_rows": len(test), "llm_calls": int(uncertain.sum()),
            **{k: round(float(v), 4) for k, v in metrics.items()},
            "accuracy_ci_low": round(lower, 4), "accuracy_ci_high": round(upper, 4),
            "weighted_rate_labels": round(truth, 4),
            "weighted_rate_model": round(observed, 4),
            "weighted_sensitivity": round(sensitivity, 4),
            "weighted_specificity": round(specificity, 4),
            "error_corrected_rate": round(estimate, 4),
        }.items()])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        test.drop(columns=["abstract"]).to_excel(xl, sheet_name="predictions", index=False)
        test[test["wrong"]].to_excel(xl, sheet_name="errors", index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
