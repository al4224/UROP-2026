"""
OpenAlex crawler for sustainability-engagement annotation.

Pulls a seeded random sample of Chemistry (field 16) and Mathematics (field 26)
articles from 2016 onwards, stratified 50/50 on whether OpenAlex's SDG
classifier tagged the work.

Writes two files per batch, next to this script:
  <prefix>_<seed>_master.xlsx      everything, including OpenAlex's SDG tags
  <prefix>_<seed>_for_coding.xlsx  blind file for annotation (no title, no tags)
Merge coding results back onto the master on `openalex_id`.

Setup, inside the activated venv:
  python -m pip install requests pandas openpyxl
  python openalex_crawler.py
"""

import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import pandas as pd
import requests
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

API_KEY = os.environ.get("OPENALEX_API_KEY")  # free key; 10x the no-key daily budget

FIELD_IDS = [16, 26]               # 16 = Chemistry, 26 = Mathematics
YEAR_FROM = 2016                   # inclusive
WORK_TYPE = "article"
LANGUAGE = "en"

N_PER_STRATUM = 150                 # 150 for real batches (300 rows)
SEED = 20260907                    # change per batch
OVERSAMPLE = 2                     # headroom for dropped and already-seen rows
MIN_ABSTRACT_CHARS = 200           # drop stubs like "No abstract available."

# Anchored to the script so a run from another directory can't start a fresh,
# empty dedup file.
HERE = Path(__file__).resolve().parent
OUT_PREFIX = "openalex_batch"
PROCESSED_IDS_FILE = HERE / "processed_ids.txt"

# Axis 1, as revised by the PI: the full phrase alone suffices; the acronym
# counts only alongside "sustainable"/"sustainability", which screens out SDG
# used as an unrelated abbreviation. The acronym ends at any non-letter rather
# than a word boundary so that numbered forms like "SDG13" still match.
SDG_PHRASE = re.compile(r"\bsustainable\s+development\s+goals?\b", re.IGNORECASE)
SDG_ACRONYM = re.compile(r"\bSDGs?(?![a-z])", re.IGNORECASE)
SUSTAIN_WORD = re.compile(r"\bsustainab(?:le|ility)\b", re.IGNORECASE)

API = "https://api.openalex.org/works"
PER_PAGE = 100                     # documented maximum
SAMPLE_MAX = 10_000                # documented maximum for `sample`
SELECT = ("id,doi,title,publication_year,abstract_inverted_index,primary_topic,"
          "sustainable_development_goals,open_access,cited_by_count")

FILTER_ANY_ABSTRACT = ",".join([
    "primary_topic.field.id:" + "|".join(map(str, FIELD_IDS)),
    f"publication_year:>{YEAR_FROM - 1}",
    f"type:{WORK_TYPE}",
    f"language:{LANGUAGE}",
])
FILTER_BASE = FILTER_ANY_ABSTRACT + ",has_abstract:true"
# OpenAlex attaches only goals scoring above 0.4, so "has any goal" is the whole
# tagged definition and there is no local threshold to apply.
FILTER_TAGGED = FILTER_BASE + ",sustainable_development_goals.id:" + "|".join(
    map(str, range(1, 18)))

MASTER_COLS = ["openalex_id", "title", "year", "abstract", "subfield", "field",
               "domain", "openalex_sdg_tag", "openalex_sdg_max_score", "stratum",
               "population_weight", "doi", "oa_status", "cited_by_count"]
# Title and SDG tags stay out: they would anchor the coder's Axis 2 judgement.
CODING_COLS = ["openalex_id", "year", "abstract", "subfield", "field", "domain"]
CODER_COLS = ["contributing_idea", "evidence_sentence", "relevant_sdg", "flagged"]

# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------


def get(params, tries=5):
    params = {**params, "api_key": API_KEY} if API_KEY else params
    for attempt in range(tries):
        try:
            r = requests.get(API, params=params, timeout=60)
        except requests.RequestException as exc:
            reason = type(exc).__name__
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code} for filter "
                                   f"{params.get('filter')}: {r.text[:400]}")
            reason = f"HTTP {r.status_code}"
        wait = 2 ** attempt
        print(f"  {reason}, retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"Gave up after {tries} attempts ({reason}). A 429 that "
                       "persists usually means the daily budget is spent.")


def count(filter_str):
    n = get({"filter": filter_str, "per_page": 1})["meta"]["count"]
    if n == 0:
        sys.exit(f"Zero works match:\n  {filter_str}\n"
                 "Check the filter syntax against the current API docs.")
    return n


def draw_sample(filter_str, n):
    """Seeded random sample, in API order, deduplicated by ID."""
    n = min(n, SAMPLE_MAX)
    works = {}
    for page in range(1, ceil(n / PER_PAGE) + 1):
        js = get({"filter": filter_str, "sample": n, "seed": SEED,
                  "per_page": PER_PAGE, "page": page, "select": SELECT})
        works.update((w["id"], w) for w in js["results"])
    return list(works.values())


# ----------------------------------------------------------------------------
# ROWS
# ----------------------------------------------------------------------------


def rebuild_abstract(inv):
    """OpenAlex ships abstracts as {token: [positions]}; invert it back."""
    words = sorted((i, tok) for tok, idxs in (inv or {}).items() for i in idxs)
    return " ".join(tok for _, tok in words)


def excel_safe(text):
    # Control characters in source metadata make openpyxl abort the whole write.
    return ILLEGAL_CHARACTERS_RE.sub("", text or "")


def short_id(url):
    return (url or "").rstrip("/").split("/")[-1]


def sdgs(work):
    goals = work.get("sustainable_development_goals") or []
    return sorted(goals, key=lambda g: -g["score"])


def sdg_keyword(abstract):
    mentions = SDG_PHRASE.search(abstract) or (
        SDG_ACRONYM.search(abstract) and SUSTAIN_WORD.search(abstract))
    return "Y" if mentions else "N"


def to_row(work):
    topic = work.get("primary_topic") or {}
    goals = sdgs(work)
    return {
        "openalex_id": short_id(work["id"]),
        "title": excel_safe(work.get("title")),
        "year": work.get("publication_year"),
        "abstract": excel_safe(rebuild_abstract(work.get("abstract_inverted_index"))),
        **{level: (topic.get(level) or {}).get("display_name")
           for level in ("subfield", "field", "domain")},
        "openalex_sdg_tag": "; ".join(
            f"{short_id(g['id'])}: {g['display_name']} ({g['score']:.2f})" for g in goals),
        "openalex_sdg_max_score": round(goals[0]["score"], 3) if goals else 0.0,
        "stratum": "tagged" if goals else "untagged",
        "doi": work.get("doi"),
        "oa_status": (work.get("open_access") or {}).get("oa_status"),
        "cited_by_count": work.get("cited_by_count"),
    }


def take(works, stratum, seen, tally):
    """
    First N_PER_STRATUM usable rows of `stratum`, in sample order. Stopping at
    the quota means drop counts cover only rows actually examined, so they
    read as rates against `examined`.
    """
    kept = []
    for row in map(to_row, works):
        if len(kept) == N_PER_STRATUM:
            break
        if row["stratum"] != stratum:
            continue
        tally["examined"] += 1
        if not row["abstract"]:
            tally["abstract_missing"] += 1
        elif len(row["abstract"]) < MIN_ABSTRACT_CHARS:
            tally["abstract_too_short"] += 1
        elif row["openalex_id"] in seen:
            tally["duplicate_earlier_batch"] += 1
        else:
            kept.append(row)
    if len(kept) < N_PER_STRATUM:
        print(f"WARNING: only {len(kept)} usable '{stratum}' rows "
              f"(wanted {N_PER_STRATUM}). Raise OVERSAMPLE.", file=sys.stderr)
    return kept


def load_processed():
    if not PROCESSED_IDS_FILE.exists():
        return set()
    return set(PROCESSED_IDS_FILE.read_text(encoding="utf-8").split())


def append_processed(ids):
    with PROCESSED_IDS_FILE.open("a", encoding="utf-8") as fh:
        fh.writelines(f"{i}\n" for i in ids)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------


def main():
    master_path = HERE / f"{OUT_PREFIX}_{SEED}_master.xlsx"
    coding_path = HERE / f"{OUT_PREFIX}_{SEED}_for_coding.xlsx"
    # A rerun with the same seed draws fresh rows past the processed ones, so
    # it would silently replace a master whose rows are already out for coding.
    if master_path.exists() or coding_path.exists():
        sys.exit(f"Batch files for SEED={SEED} already exist. Change SEED.")
    if not API_KEY:
        print("OPENALEX_API_KEY not set: using the smaller no-key budget.",
              file=sys.stderr)
    seen = load_processed()
    print(f"Previously processed IDs on file: {len(seen)}")

    pop_any_abstract = count(FILTER_ANY_ABSTRACT)
    pop_total = count(FILTER_BASE)
    pop_tagged = count(FILTER_TAGGED)
    pop_untagged = pop_total - pop_tagged
    if pop_untagged <= 0:
        sys.exit("Tagged count is not below the total; the SDG filter is not "
                 "behaving as documented.")
    print(f"\nPopulation, any abstract status: {pop_any_abstract:,}")
    print(f"Population, with abstract:       {pop_total:,}")
    print(f"  tagged:   {pop_tagged:,}\n  untagged: {pop_untagged:,}")

    want = N_PER_STRATUM * OVERSAMPLE
    print("\nDrawing tagged stratum...")
    tagged_draw = draw_sample(FILTER_TAGGED, want)
    if not all(map(sdgs, tagged_draw)):
        sys.exit("The tagged filter returned works with no SDG goals attached.")

    # No documented filter selects works with zero goals, so untagged rows come
    # from a base-population sample sized to hold `want` of them on average.
    print("Drawing base-population pool for the untagged stratum...")
    pool = draw_sample(FILTER_BASE, ceil(want * pop_total / pop_untagged))
    pool_tagged_share = sum(bool(sdgs(w)) for w in pool) / len(pool)
    print(f"  pool of {len(pool)}: tagged share {pool_tagged_share:.1%} "
          f"(population {pop_tagged / pop_total:.1%})")

    tally = Counter()
    rows = take(tagged_draw, "tagged", seen, tally) + take(pool, "untagged", seen, tally)
    if not rows:
        sys.exit("No usable rows. Check filters.")
    # Shuffled so row order can't reveal the stratum to the coder.
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED)

    # Sampling is 50/50 by design, so a raw rate over this file is NOT a
    # population rate. Weight each row by population share / sample share.
    stratum = df["stratum"]
    pop_share = stratum.map({"tagged": pop_tagged / pop_total,
                             "untagged": pop_untagged / pop_total})
    sample_share = stratum.map(stratum.value_counts(normalize=True))
    df["population_weight"] = (pop_share / sample_share).round(4)

    sampled = stratum.value_counts()
    meta = pd.DataFrame([
        ("run_timestamp_utc", datetime.now(timezone.utc).isoformat()),
        ("seed", SEED),
        ("filter_base", FILTER_BASE),
        ("filter_tagged", FILTER_TAGGED),
        ("corpus", "core (API default)"),
        ("oa_filter_applied", "none (abstract presence only)"),
        ("sdg_tag_definition", "any goal attached by OpenAlex (its cut-off: score > 0.4)"),
        ("stratum_split_method", "tagged: API filter on goals 1-17; "
                                 "untagged: local split of a base-population sample"),
        ("population_without_abstract_filter", pop_any_abstract),
        ("population_total", pop_total),
        ("population_tagged", pop_tagged),
        ("population_untagged", pop_untagged),
        ("pool_size", len(pool)),
        ("pool_tagged_share", round(pool_tagged_share, 4)),
        ("sampled_tagged", int(sampled.get("tagged", 0))),
        ("sampled_untagged", int(sampled.get("untagged", 0))),
        ("rows_examined", tally["examined"]),
        ("dropped_abstract_missing", tally["abstract_missing"]),
        ("dropped_abstract_too_short", tally["abstract_too_short"]),
        ("dropped_duplicate_earlier_batch", tally["duplicate_earlier_batch"]),
    ], columns=["key", "value"])

    coding = df[CODING_COLS].assign(
        sdg_keyword=df["abstract"].map(sdg_keyword),
        **dict.fromkeys(CODER_COLS, ""),
    )

    with pd.ExcelWriter(master_path, engine="openpyxl") as xl:
        df[MASTER_COLS].to_excel(xl, sheet_name="works", index=False)
        meta.to_excel(xl, sheet_name="sampling_metadata", index=False)
    coding.to_excel(coding_path, index=False)

    append_processed(df["openalex_id"])

    print("\n" + "=" * 60)
    print(f"Rows written:          {len(df)} ({sampled.get('tagged', 0)} tagged / "
          f"{sampled.get('untagged', 0)} untagged)")
    print(f"Rows examined:         {tally['examined']}")
    print(f"Dropped, no abstract:  {tally['abstract_missing']}")
    print(f"Dropped, too short:    {tally['abstract_too_short']}")
    print(f"Dropped, seen before:  {tally['duplicate_earlier_batch']}")
    print(f"\nMaster:  {master_path}")
    print(f"Coding:  {coding_path}   <- send this one for annotation")
    print("=" * 60)


if __name__ == "__main__":
    main()