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
import time

import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score

from ml_baseline import DATA, load, score

MODEL = "qwen3.6:latest"          # any recent instruct model, 7B-14B, fits a 12 GB card
OLLAMA = "http://localhost:11434/api/generate"
CACHE = DATA / "ml_llm_cache.json"
SCORED = DATA / "ml_errors_embed.xlsx"        # written by ml_errors.py embed
OUT = DATA / "ml_llm_predictions.xlsx"
UNCERTAIN = (0.35, 0.65)
# Triage only needs the LLM inside the uncertain band, which is about a third of
# the rows. Set to False to judge everything and score the LLM on its own.
ONLY_UNCERTAIN = True
TIMEOUT = 600          # the first call also loads the model into memory
MAX_TOKENS = 400       # the answer is one short JSON object; stop runaway output
THINKING = False       # reasoning models otherwise spend minutes before answering

# Condensed from annotation_ruleset_v3, sections 3 to 9. Keep it faithful: this
# is the coder's rule, not a paraphrase to tune. Update it when the ruleset moves.
PROMPT = """You apply a fixed annotation rule to a research abstract from chemistry \
or mathematics.

THE TEST, in two steps:
1. Does the abstract name a REAL-WORLD THING - a domain, system, application or \
problem - rather than only objects of study (equations, structures, datasets, \
idealised models)?
2. Does that thing stand in a recognised relationship to any UN Sustainable \
Development Goal area, statable in one step, without supplying a use the abstract \
never mentions?

Both yes -> Y. Otherwise -> N.

WHAT DOES NOT MATTER: the format of the work (software, guidelines, reviews, \
editorials and empirical studies are judged identically); the depth of treatment (a \
bare field-of-interest label, one item in a list, a motivating sentence or a \
speculative future use all count); sustainability language (neither required nor \
sufficient). ONE qualifying domain is enough.

QUALIFYING DOMAINS:
- Health and medicine: drug discovery, pharmaceuticals, therapeutics, clinical \
trials, diagnosis, epidemiology, public health, named diseases, generic references \
to disease or clinical conditions. Drug discovery, medicinal chemistry and \
pharmacology count IN ANY FORM.
- Education: classroom interventions, teacher training, student achievement, \
curriculum, learning materials, assessment.
- Food and agriculture: crops, nutrition, food safety and authenticity, \
agrochemistry, herbicides, farming practice.
- Water and oceans: water quality, drinking water, groundwater, coastal systems, \
water treatment.
- Energy: storage, batteries, hydrogen, fuels, electrocatalysis, renewables, \
energy-efficiency gains.
- Climate: greenhouse gases, emissions, urban heat, atmospheric systems.
- Pollution and toxics: contaminants, heavy metals, toxic gases, wastewater \
pollutants.
- Materials and waste: recycling, waste valorisation, resource recovery.
- Ecosystems: biodiversity, habitats, forestry, species.
- Governance and institutions: fraud detection, data manipulation, transparency, \
access to medicines, patent or monopoly effects.
- Economy and work: macroeconomic activity, employment outcomes, human capital, \
smallholder livelihoods.
- Social equity: demographic disparities by race, sex or income.
- Green chemistry and physical systems: see the two special rules below.

GREEN CHEMISTRY counts (Y): a named green solvent or water as solvent; \
solvent-free or mechanochemical; recyclable or reused catalyst, reagent or \
auxiliary; air or oxygen as oxidant; metal-free or oxidant-free stated as a \
feature; specified mild conditions, low pressure or room temperature; one-pot or \
pot economy; atom or step economy; visible-light or photochemical driving of a \
synthesis; eliminating a purification step such as avoiding column chromatography; \
the explicit words green, benign, clean, waste or sustainable.
It does NOT count on generic yield or convenience language alone: "efficient", \
"good yields", "high selectivity", "operationally simple", "readily available \
substrates", "bench-stable".

PHYSICAL SYSTEMS: a named EQUATION CLASS is not a named PHYSICAL SYSTEM. \
"Parabolic equations", "nonlinear Schrodinger equations", "convection-diffusion \
problem" are mathematical objects -> N. "Shallow-water waves", "turbulent channel \
flow", "groundwater transport" name something real -> Y. A real physical system \
must still sit plausibly near climate, energy, water or another goal area: \
rarefied gas venting into a vacuum is genuine gas flow but too far from any goal \
-> N. Idealised models (Ising, harmonic oscillator, ASEP, random matrices) -> N, \
and "flow" in its differential-geometry sense -> N.

INDUSTRIAL WORK is neither qualifying nor disqualifying. Ask only whether the \
specific application points toward an SDG outcome: does it make something cleaner, \
more efficient, less wasteful or more accessible, or is it simply making a \
product? Ethylene purification, lubricant thermal stability, biomanufacturing \
yield and jet fuel contaminant removal are Y. Commodity plastics production, \
photo-RDRP for advanced manufacturing, industrial heat treatment and military \
explosives are N.

REMAINS N: general-purpose methods, tools and statistical tutorials naming no \
domain; fundamental molecular or structural biology with no stated application \
(protein structure prediction, sequence alignment, cell atlases); pure synthetic \
or structural chemistry with no application and no green-chemistry descriptor; \
astronomy and cosmology; technology named as a bare capability with no one-step \
goal link (AI robustness, encryption, sensing, scintillators); photocatalysis \
named as a material property with no stated use; "biologically significant" or \
"biologically active" with no named activity, target or disease; nanomaterial \
biodegradation framed purely as a biochemical finding; bare "economic \
applications" as a methodological justification.

SETTLED PRECEDENTS: model-organism plant under temperature stress -> Y; \
financial-statement or government-data fraud detection -> Y; social-science use \
naming race, sex or education -> Y; statistics demonstrated on epidemiological or \
clinical-trial data -> Y; statistics with no health context (survival analysis, \
sample size) -> N; antioxidant chemistry naming a food or health application -> Y, \
naming none -> N; nuclear-analogue geochemistry with no waste framing -> N.

ABSTRACT:
{abstract}

Answer with JSON only, no other text:
{{"label": "Y" or "N", "sentence": "the verbatim sentence that decides it, or \
empty if N"}}"""


def ask(abstract, think=THINKING):
    payload = {
        "model": MODEL,
        "prompt": PROMPT.format(abstract=abstract),
        "stream": False,
        "options": {"temperature": 0, "num_predict": MAX_TOKENS},
    }
    if think is not None:
        payload["think"] = think
    response = requests.post(OLLAMA, timeout=TIMEOUT, json=payload)
    if response.status_code == 400 and "think" in response.text.lower():
        return ask(abstract, think=None)      # model has no thinking mode to turn off
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
    started = time.time()
    for n, row in enumerate(todo, 1):
        label, sentence = parse(ask(row.abstract))
        cache[row.openalex_id] = {"label": label, "sentence": sentence}
        if n == 1:
            each = time.time() - started
            print(f"  first answer in {each:.0f}s (includes loading the model); "
                  f"{len(todo)} rows is roughly {each * len(todo) / 60:.0f} min at that rate")
        if n % 10 == 0 or n == len(todo):
            CACHE.write_text(json.dumps(cache))
            print(f"  {n}/{len(todo)}  ({(time.time() - started) / n:.1f}s per row)")
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

    low, high = UNCERTAIN
    uncertain = dev["score"].between(low, high)
    asked = dev[uncertain] if ONLY_UNCERTAIN else dev

    answers = judge(asked)
    dev["llm_label"] = [answers.get(i, {}).get("label", "") for i in dev["openalex_id"]]
    dev["llm_sentence"] = [answers.get(i, {}).get("sentence", "") for i in dev["openalex_id"]]
    unreadable = int(((dev["llm_label"] == "") & dev["openalex_id"].isin(answers)).sum())
    if unreadable:
        print(f"WARNING: {unreadable} replies could not be parsed; counted as N")
    dev["llm_pred"] = (dev["llm_label"] == "Y").astype(int)

    if ONLY_UNCERTAIN:
        report("LLM, uncertain band only", dev.loc[uncertain, "y"], dev.loc[uncertain, "llm_pred"])
        report("Embeddings, same rows", dev.loc[uncertain, "y"], dev.loc[uncertain, "embed_pred"])
    else:
        report("LLM alone", dev["y"], dev["llm_pred"])
    report("Embeddings alone", dev["y"], dev["embed_pred"])

    combined = dev["embed_pred"].where(~uncertain, dev["llm_pred"])
    report(f"Triage: embeddings outside {low}-{high}, LLM inside", dev["y"], combined)
    print(f"  ({uncertain.sum()} of {len(dev)} rows sent to the LLM)")

    both = dev[dev["openalex_id"].isin(answers)]
    print("\nAgreement between the two models, on the rows asked: "
          f"{(both['llm_pred'] == both['embed_pred']).mean():.1%} "
          f"(kappa {cohen_kappa_score(both['llm_pred'], both['embed_pred']):.3f})")

    dev.to_excel(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()