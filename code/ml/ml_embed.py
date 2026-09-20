"""
Semantic embeddings for the Axis 2 judgement.

Word-frequency models only recognise vocabulary they have seen. Embeddings map
text into a space where "antiviral activity against RNA viruses" sits near the
health examples already in the training data, which is where the baseline's
missed Ys came from.

Encodes once and caches, then scores the same two ways as the baseline, using
the same folds so the numbers are comparable:

  abstract_embed  one vector per abstract
  sentence_embed  one vector per sentence; Y if any sentence scores Y

First run downloads the model (about 90 MB for the default).

Run:  python code/ml/ml_embed.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ml_baseline import DATA, FOLDS, SEED, load, score

# Swapping this is the main knob. all-mpnet-base-v2 scored best on the dev
# rows; all-MiniLM-L6-v2 is smaller and faster for a quick check. Each model
# caches its vectors separately, so switching back costs nothing.
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CACHE = DATA / f"ml_embeddings_{MODEL_NAME.split('/')[-1]}.npz"
OUT = DATA / "ml_eval_embed.xlsx"
BATCH = 64


def encode(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"Encoding {len(texts)} texts on {model.device}...")
    return model.encode(texts, batch_size=BATCH, normalize_embeddings=True,
                        show_progress_bar=True)


def embeddings(keys, texts):
    """Encode only what is not already cached; keys identify each text."""
    stored = {}
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)
        stored = dict(zip(cached["keys"].tolist(), cached["vectors"]))
    missing = [(k, t) for k, t in zip(keys, texts) if k not in stored]
    if missing:
        vectors = encode([t for _, t in missing])
        stored.update({k: v for (k, _), v in zip(missing, vectors)})
        np.savez(CACHE, keys=np.array(list(stored)),
                 vectors=np.stack(list(stored.values())))
        print(f"Cached {len(stored)} vectors in {CACHE.name}")
    return np.stack([stored[k] for k in keys])


def classifier():
    return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)


def abstract_scores(train, test, data):
    model = classifier().fit(data["abstract_x"][train.index], train["y"])
    return pd.Series(model.predict_proba(data["abstract_x"][test.index])[:, 1],
                     index=test.index)


def sentence_scores(train, test, data):
    """Train on labelled sentences, score an abstract by its strongest one."""
    sentences, vectors = data["sentences"], data["sentence_x"]
    known = (sentences["sentence_label"].isin([0, 1])
             & sentences["openalex_id"].isin(train["openalex_id"]))
    model = classifier().fit(vectors[known.values],
                             sentences.loc[known, "sentence_label"])

    wanted = sentences["openalex_id"].isin(test["openalex_id"])
    scored = sentences[wanted].assign(
        p=model.predict_proba(vectors[wanted.values])[:, 1])
    best = scored.groupby("openalex_id")["p"].max()
    return pd.Series([best.get(i, 0.0) for i in test["openalex_id"]], index=test.index)


MODELS = {"abstract_embed": abstract_scores, "sentence_embed": sentence_scores}


def prepare_data(dev, sentences):
    """Embeddings for every abstract and sentence, encoded once and cached."""
    return {
        "sentences": sentences,
        "abstract_x": embeddings([f"{i}:abstract" for i in dev["openalex_id"]],
                                 dev["abstract"].tolist()),
        "sentence_x": embeddings(
            [f"{i}:{n}" for i, n in zip(sentences["openalex_id"], sentences["sentence_index"])],
            sentences["sentence"].tolist()),
    }


def main():
    dev, sentences = load()
    dev["y"] = (dev["label"] == "Y").astype(int)
    sentences = sentences[sentences["openalex_id"].isin(dev["openalex_id"])].reset_index(drop=True)

    data = prepare_data(dev, sentences)

    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    results = []
    for name, score_fold in MODELS.items():
        for fold, (train_idx, test_idx) in enumerate(folds.split(dev, dev["y"])):
            train, test = dev.iloc[train_idx], dev.iloc[test_idx]
            predicted = (score_fold(train, test, data) >= 0.5).astype(int)
            results.append({"model": name, "fold": fold, **score(test["y"], predicted)})

    results = pd.DataFrame(results)
    summary = results.groupby("model").mean(numeric_only=True).drop(columns="fold").round(3)
    print(f"\n{MODEL_NAME}, {len(dev)} dev rows\n")
    print(summary.to_string())

    baseline = DATA / "ml_eval_baseline.xlsx"
    if baseline.exists():
        previous = pd.read_excel(baseline, sheet_name="folds")
        print("\nBaselines for comparison:")
        print(previous.groupby("model").mean(numeric_only=True)
              .drop(columns="fold").round(3).to_string())

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        results.to_excel(xl, sheet_name="folds", index=False)
        summary.to_excel(xl, sheet_name="summary")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
