"""
Baselines for the Axis 2 judgement, scored by 5-fold cross-validation on the
dev rows. Nothing here is the final model: these set the floor that a real
model has to beat, and show whether the task is mostly keyword matching.

  majority        always predict the commoner class
  lexicon         Y if the abstract mentions a domain from ruleset section 4
  abstract_tfidf  word frequencies over the whole abstract
  sentence_tfidf  word frequencies per sentence; Y if any sentence scores Y

Run:  python code/ml/ml_baseline.py
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).resolve().parents[2] / "data"
DATASET = DATA / "ml_dataset.xlsx"
OUT = DATA / "ml_eval_baseline.xlsx"
SEED = 0
FOLDS = 5

# Domain vocabulary drawn from ruleset section 4. Deliberately crude: this is a
# floor to beat, and a measure of how far plain keywords get.
LEXICON = re.compile(r"""\b(
 drug|drugs|pharmaceutic\w*|therapeut\w*|clinic\w*|diagnos\w*|prognos\w*|disease\w*|
 patholog\w*|cancer|tumou?r\w*|patient\w*|antibacterial|antimicrobial|antiviral|health|
 agricultur\w*|crop\w*|food|nutrition\w*|pesticide\w*|fertilis\w*|fertiliz\w*|
 water|groundwater|wastewater|coastal|ocean\w*|marine|desalinat\w*|
 energy|batter\w*|capacitor\w*|photovoltaic\w*|solar|fuel\w*|electrocataly\w*|hydrogen|
 greenhouse|emission\w*|climate|carbon\s+dioxide|co2|atmospher\w*|
 pollut\w*|contaminant\w*|toxic\w*|waste\w*|remediat\w*|
 recycl\w*|plastic\w*|biodegrad\w*|sustainab\w*|green\s+chemistry|solvent\s+free|
 biodivers\w*|habitat\w*|species|ecosystem\w*|
 turbulen\w*|fluid\s+flow|heat\s+conduction|diffusion|wave\s+propagation
)\b""", re.IGNORECASE | re.VERBOSE)


def load():
    abstracts = pd.read_excel(DATASET, sheet_name="abstracts", keep_default_na=False)
    sentences = pd.read_excel(DATASET, sheet_name="sentences", keep_default_na=False)
    dev = abstracts[abstracts["split"] == "dev"].reset_index(drop=True)
    return dev, sentences


def tfidf_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))


def predict_majority(train, test, _sentences):
    winner = train["y"].mode()[0]
    return [winner] * len(test)


def predict_lexicon(_train, test, _sentences):
    return [int(bool(LEXICON.search(t))) for t in test["abstract"]]


def predict_abstract_tfidf(train, test, _sentences):
    model = tfidf_model().fit(train["abstract"], train["y"])
    return model.predict(test["abstract"])


def predict_sentence_tfidf(train, test, sentences):
    """Train on single sentences, then judge an abstract by its best sentence."""
    known = sentences[sentences["openalex_id"].isin(train["openalex_id"])]
    known = known[known["sentence_label"].isin([0, 1])]
    model = tfidf_model().fit(known["sentence"], known["sentence_label"])

    scored = sentences[sentences["openalex_id"].isin(test["openalex_id"])].copy()
    scored["p"] = model.predict_proba(scored["sentence"])[:, 1]
    best = scored.groupby("openalex_id")["p"].max()
    return [int(best.get(i, 0.0) >= 0.5) for i in test["openalex_id"]]


MODELS = {
    "majority": predict_majority,
    "lexicon": predict_lexicon,
    "abstract_tfidf": predict_abstract_tfidf,
    "sentence_tfidf": predict_sentence_tfidf,
}


def score(truth, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=[1], zero_division=0)
    return {"accuracy": (truth == predicted).mean(), "precision_Y": precision[0],
            "recall_Y": recall[0], "f1_Y": f1[0],
            "kappa": cohen_kappa_score(truth, predicted)}


def cross_validate(dev, sentences):
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    results = []
    for name, predict in MODELS.items():
        for fold, (train_idx, test_idx) in enumerate(folds.split(dev, dev["y"])):
            train, test = dev.iloc[train_idx], dev.iloc[test_idx]
            predicted = pd.Series(predict(train, test, sentences), index=test.index)
            results.append({"model": name, "fold": fold, **score(test["y"], predicted)})
    return pd.DataFrame(results)


def learning_curve(dev, sentences, sizes=(50, 100, 150, 200, 236)):
    """How much would more labelled data buy? Train on subsets, test on a fixed fold."""
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for size in sizes:
        for fold, (train_idx, test_idx) in enumerate(folds.split(dev, dev["y"])):
            train = dev.iloc[train_idx]
            if size > len(train):
                continue
            subset = train.sample(size, random_state=SEED + fold)
            test = dev.iloc[test_idx]
            predicted = predict_abstract_tfidf(subset, test, sentences)
            rows.append({"train_rows": size, "fold": fold,
                         "accuracy": (test["y"] == predicted).mean()})
    return pd.DataFrame(rows).groupby("train_rows")["accuracy"].agg(["mean", "std"])


def main():
    dev, sentences = load()
    dev["y"] = (dev["label"] == "Y").astype(int)
    print(f"dev rows: {len(dev)}  ({dev['y'].sum()} Y / {len(dev) - dev['y'].sum()} N)\n")

    results = cross_validate(dev, sentences)
    summary = results.groupby("model").agg(["mean", "std"]).round(3)
    summary = summary[[(m, s) for m in ["accuracy", "precision_Y", "recall_Y", "f1_Y", "kappa"]
                       for s in ["mean", "std"]]]
    print(summary.to_string())

    print("\nLearning curve (abstract_tfidf):")
    curve = learning_curve(dev, sentences)
    print(curve.round(3).to_string())

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        results.to_excel(xl, sheet_name="folds", index=False)
        summary.to_excel(xl, sheet_name="summary")
        curve.to_excel(xl, sheet_name="learning_curve")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
