"""
Local LLM judge for the Axis 2 question.

The embedding model recognises vocabulary; it does not check whether a
real-world domain is actually named. This asks a local model to apply the rule
itself, and measures it two ways:

  standalone  the LLM judging every dev row
  triage      embeddings where confident, LLM only in the uncertain band

Needs Ollama running locally (https://ollama.com) with a model pulled:
    ollama pull qwen2.5:14b

Answers are cached in data/ml_llm_cache.json, so a rerun only asks about rows
it has not seen. Delete that file after changing MODEL or PROMPT.

Run:  python code/ml/ml_llm.py
"""

import json
import re
import sys

import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score

from ml_baseline import DATA, load, score

MODEL = "qwen3.6"          # any recent instruct model, 7B-14B, fits a 12 GB card
OLLAMA = "http://localhost:11434/api/generate"
CACHE = DATA / "ml_llm_cache.json"
SCORED = DATA / "ml_errors_embed.xlsx"        # written by ml_errors.py embed
OUT = DATA / "ml_llm_predictions.xlsx"
UNCERTAIN = (0.35, 0.65)
TIMEOUT = 180

# Condensed from annotation_ruleset_v2 sections 3 to 5. Keep it faithful: this
# is the coder's rule, not a paraphrase to tune.
PROMPT = """You are applying a fixed annotation rule to a research abstract.

THE TEST: Does the abstract name any real-world domain that connects to a UN \
Sustainable Development Goal area?

Two steps:
1. Does the abstract name a real-world thing - a domain, system, application or \
problem - as opposed to only objects of study (sequences, structures, datasets, \
equations, idealised models)?
2. Does that thing stand in a recognised relationship to an SDG area, statable in \
one step, without supplying a downstream use the abstract never mentions?

Both steps yes -> Y. Otherwise -> N.

Qualifying domain areas include: health and medicine (drug discovery, therapeutics, \
clinical application, diagnosis, named diseases, generic references to disease or \
clinical conditions); food and agriculture; water and oceans; energy (storage, \
batteries, electrocatalysis, fuels, renewables); climate (greenhouse gases, \
emissions); pollution and toxics; green chemistry (cleaner synthesis, waste or \
solvent reduction, atom economy, mild conditions, renewable inputs); materials and \
waste (recycling, plastics, resource recovery); ecosystems; physical systems \
treated as real (fluid flow, turbulence, heat conduction, diffusion, wave \
propagation); explicit low-resource or capacity-building framing.

What does NOT matter: the format of the work (software, reporting standard, \
review and empirical study are judged identically); the depth of treatment (a \
single motivating sentence or one item in a list of applications counts as much as \
the whole paper being about it); sustainability language (neither required nor \
sufficient). One qualifying domain is enough.

What remains N: general-purpose methods and tools with no stated domain; \
fundamental molecular or structural biology naming no applied domain; idealised or \
toy systems; fundamental chemistry with no application and no efficiency or \
cleanliness claim.

ABSTRACT:
{abstract}

Answer with JSON only, no other text:
{{"label": "Y" or "N", "sentence": "the verbatim sentence that decides it, or \
empty if N"}}"""


def ask(abstract):
    response = requests.post(OLLAMA, timeout=TIMEOUT, json={
        "model": MODEL,
        "prompt": PROMPT.format(abstract=abstract),
        "stream": False,
        "options": {"temperature": 0},
    })
    response.raise_for_status()
    return response.json()["response"]


def parse(reply):
    """The model is asked for JSON; accept it wrapped in anything else."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return "", ""
    try:
        answer = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", ""
    label = str(answer.get("label", "")).strip().upper()
    return (label if label in ("Y", "N") else ""), str(answer.get("sentence", ""))


def judge(rows):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [r for r in rows.itertuples() if r.openalex_id not in cache]
    print(f"{len(cache)} cached, {len(todo)} to ask")
    for n, row in enumerate(todo, 1):
        label, sentence = parse(ask(row.abstract))
        cache[row.openalex_id] = {"label": label, "sentence": sentence}
        if n % 10 == 0 or n == len(todo):
            CACHE.write_text(json.dumps(cache))
            print(f"  {n}/{len(todo)}")
    CACHE.write_text(json.dumps(cache))
    return cache


def report(name, truth, predicted):
    print(f"\n{name}: {len(truth)} rows")
    print(pd.Series(score(truth.values, predicted.values)).round(3).to_string())


def main():
    dev, _ = load()
    dev["y"] = (dev["label"] == "Y").astype(int)

    if not SCORED.exists():
        sys.exit(f"Missing {SCORED.name}. Run: python code/ml/ml_errors.py embed")
    scored = pd.read_excel(SCORED, sheet_name="all_scored", keep_default_na=False)
    dev = dev.merge(scored[["openalex_id", "score", "predicted"]]
                    .rename(columns={"predicted": "embed_pred"}), on="openalex_id")

    answers = judge(dev)
    dev["llm_label"] = [answers[i]["label"] for i in dev["openalex_id"]]
    dev["llm_sentence"] = [answers[i]["sentence"] for i in dev["openalex_id"]]
    unreadable = (dev["llm_label"] == "").sum()
    if unreadable:
        print(f"WARNING: {unreadable} replies could not be parsed; counted as N")
    dev["llm_pred"] = (dev["llm_label"] == "Y").astype(int)

    report("LLM alone", dev["y"], dev["llm_pred"])
    report("Embeddings alone", dev["y"], dev["embed_pred"])

    low, high = UNCERTAIN
    uncertain = dev["score"].between(low, high)
    combined = dev["embed_pred"].where(~uncertain, dev["llm_pred"])
    report(f"Triage: embeddings outside {low}-{high}, LLM inside", dev["y"], combined)
    print(f"  ({uncertain.sum()} of {len(dev)} rows sent to the LLM)")

    print("\nAgreement between the two models: "
          f"{(dev['llm_pred'] == dev['embed_pred']).mean():.1%} "
          f"(kappa {cohen_kappa_score(dev['llm_pred'], dev['embed_pred']):.3f})")

    dev.to_excel(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
