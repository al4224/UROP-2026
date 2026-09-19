#!/usr/bin/env python3
"""
OpenAlex crawler for sustainability-engagement annotation.

Pulls a seeded random sample of Mathematics (field 26) and Chemistry (field 16)
journal articles from 2016 onwards, stratified 50/50 on whether OpenAlex's SDG
classifier assigned the work any tag.

Outputs two files:
  1. <prefix>_master.xlsx    -- everything, including the OpenAlex SDG tags.
  2. <prefix>_for_coding.xlsx -- the same rows with title and SDG tags REMOVED,
     so the human/model coder cannot be anchored by OpenAlex's own judgement.
     Merge results back onto the master on `openalex_id`.

Requires: pip install requests pandas openpyxl
"""

import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

EMAIL = "al4224@ic.ac.uk"          # REQUIRED: puts you in the polite pool
API_KEY = os.environ.get("OPENALEX_API_KEY")  # optional, raises daily budget 10x

FIELD_IDS = [16, 26]               # 16 = Chemistry, 26 = Mathematics
YEAR_FROM = 2016                   # inclusive
WORK_TYPE = "article"
LANGUAGE = "en"

N_PER_STRATUM = 10                 # final rows per stratum -> 300 total
SEED = 20260910                    # change per batch; record it in your notes
POOL_MULTIPLIER = 6                # oversample factor for the fallback splitter

# An SDG tag counts as "present" only if its prediction score clears this bar.
# The classifier is generous and assigns low-confidence tags very widely, so
# 0.0 would put almost everything in the tagged stratum. Inspect the score
# distribution printed at the end of the run before settling on a value.
SDG_SCORE_THRESHOLD = 0.40

MIN_ABSTRACT_CHARS = 200           # drop stubs like "No abstract available."
OUT_PREFIX = "openalex_batch"
PROCESSED_IDS_FILE = "processed_ids.txt"  # cumulative; prevents cross-batch dupes

BASE = "https://api.openalex.org/works"
SELECT = ",".join([
    "id", "doi", "title", "publication_year", "type", "language",
    "abstract_inverted_index", "primary_topic",
    "sustainable_development_goals", "open_access", "cited_by_count",
])

# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


def _params(extra):
    p = {"mailto": EMAIL}
    if API_KEY:
        p["api_key"] = API_KEY
    p.update(extra)
    return p


def get(url, params, tries=5):
    """GET with exponential backoff. OpenAlex is generous but not infinite."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=_params(params), timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                print(f"  HTTP {r.status_code}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  {type(exc).__name__}, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {tries} attempts: {url} {params}")


def base_filter(extra=None):
    parts = [
        "primary_topic.field.id:" + "|".join(f"fields/{f}" for f in FIELD_IDS),
        f"publication_year:>{YEAR_FROM - 1}",
        f"type:{WORK_TYPE}",
        f"language:{LANGUAGE}",
        "has_abstract:true",
    ]
    if extra:
        parts.append(extra)
    return ",".join(parts)


def count_for(filter_str):
    js = get(BASE, {"filter": filter_str, "per_page": 1})
    return js["meta"]["count"]


# ----------------------------------------------------------------------------
# ABSTRACT RECONSTRUCTION
# ----------------------------------------------------------------------------


def rebuild_abstract(inv):
    """OpenAlex ships abstracts as {token: [positions]}. Invert it back."""
    if not inv:
        return None
    positions = []
    for token, idxs in inv.items():
        for i in idxs:
            positions.append((i, token))
    if not positions:
        return None
    positions.sort()
    return " ".join(tok for _, tok in positions).strip()


# ----------------------------------------------------------------------------
# SDG HANDLING
# ----------------------------------------------------------------------------


def sdg_list(work):
    return work.get("sustainable_development_goals") or []


def max_sdg_score(work):
    scores = [s.get("score", 0.0) or 0.0 for s in sdg_list(work)]
    return max(scores) if scores else 0.0


def is_tagged(work):
    return max_sdg_score(work) >= SDG_SCORE_THRESHOLD


def format_sdgs(work):
    """'6: Clean water and sanitation (0.81); 13: Climate action (0.44)'"""
    out = []
    for s in sorted(sdg_list(work), key=lambda x: -(x.get("score") or 0)):
        sid = (s.get("id") or "").rstrip("/").split("/")[-1]
        out.append(f"{sid}: {s.get('display_name')} ({s.get('score'):.2f})")
    return "; ".join(out) if out else ""


# ----------------------------------------------------------------------------
# SAMPLING
# ----------------------------------------------------------------------------


def draw_sample(filter_str, n, seed):
    """Seeded random sample via OpenAlex's own `sample` parameter (max 10k)."""
    if n > 10000:
        raise ValueError("OpenAlex caps `sample` at 10,000 records.")
    rows, page, per_page = [], 1, 200
    while len(rows) < n:
        js = get(BASE, {
            "filter": filter_str,
            "sample": n,
            "seed": seed,
            "per_page": min(per_page, n - len(rows)),
            "page": page,
            "select": SELECT,
        })
        batch = js.get("results", [])
        if not batch:
            break
        rows.extend(batch)
        page += 1
        time.sleep(0.15)
    return rows[:n]


def try_null_filter():
    """
    Ask the API directly for works with no SDG tag. Support for `:null` on this
    field is not guaranteed, so probe it and fall back if it errors or returns
    an implausible count.
    """
    try:
        untagged = count_for(base_filter("sustainable_development_goals.id:null"))
        total = count_for(base_filter())
        if 0 < untagged < total:
            return untagged, total
    except Exception as exc:
        print(f"  null-filter probe failed ({exc}); using fallback splitter",
              file=sys.stderr)
    return None, None


# ----------------------------------------------------------------------------
# ROW ASSEMBLY
# ----------------------------------------------------------------------------


def to_row(work, stratum):
    topic = work.get("primary_topic") or {}
    return {
        "openalex_id": (work.get("id") or "").rstrip("/").split("/")[-1],
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "abstract": rebuild_abstract(work.get("abstract_inverted_index")),
        "subfield": (topic.get("subfield") or {}).get("display_name"),
        "field": (topic.get("field") or {}).get("display_name"),
        "domain": (topic.get("domain") or {}).get("display_name"),
        "openalex_sdg_tag": format_sdgs(work),
        "openalex_sdg_max_score": round(max_sdg_score(work), 3),
        "stratum": stratum,
        "doi": work.get("doi"),
        "oa_status": (work.get("open_access") or {}).get("oa_status"),
        "cited_by_count": work.get("cited_by_count"),
    }


def clean(rows, drops):
    """Drop rows whose abstract failed to reconstruct or is too short."""
    kept = []
    for r in rows:
        a = r["abstract"]
        if not a:
            drops["abstract_missing_despite_filter"] += 1
            continue
        if len(a) < MIN_ABSTRACT_CHARS:
            drops["abstract_too_short"] += 1
            continue
        kept.append(r)
    return kept


def load_processed():
    if not os.path.exists(PROCESSED_IDS_FILE):
        return set()
    with open(PROCESSED_IDS_FILE) as fh:
        return {line.strip() for line in fh if line.strip()}


def append_processed(ids):
    with open(PROCESSED_IDS_FILE, "a") as fh:
        for i in ids:
            fh.write(i + "\n")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------


def main():
    if EMAIL == "you@example.com":
        sys.exit("Set EMAIL at the top of the script before running.")

    random.seed(SEED)
    drops = Counter()
    already = load_processed()
    print(f"Previously processed IDs on file: {len(already)}")

    # --- population counts, needed later for reweighting -------------------
    total_pop = count_for(base_filter())
    print(f"\nPopulation matching filters: {total_pop:,}")
    if total_pop == 0:
        sys.exit("Filter returned zero works. Check the field ID syntax "
                 "('fields/16') against the current API before going further.")

    untagged_pop, _ = try_null_filter()
    use_null_filter = untagged_pop is not None
    if use_null_filter:
        tagged_pop = total_pop - untagged_pop
        print(f"  with SDG tag:    {tagged_pop:,}")
        print(f"  without SDG tag: {untagged_pop:,}")

    # --- draw the two strata ----------------------------------------------
    if use_null_filter:
        print("\nDrawing tagged stratum...")
        tagged_raw = draw_sample(
            base_filter("sustainable_development_goals.id:!null"),
            N_PER_STRATUM * 2, SEED)
        print("Drawing untagged stratum...")
        untagged_raw = draw_sample(
            base_filter("sustainable_development_goals.id:null"),
            N_PER_STRATUM * 2, SEED)
        tagged = [to_row(w, "tagged") for w in tagged_raw]
        untagged = [to_row(w, "untagged") for w in untagged_raw]
    else:
        # Fallback: one big seeded draw, split locally on the score threshold.
        pool_n = min(10000, N_PER_STRATUM * 2 * POOL_MULTIPLIER)
        print(f"\nDrawing pooled sample of {pool_n} and splitting locally...")
        pool = draw_sample(base_filter(), pool_n, SEED)
        tagged = [to_row(w, "tagged") for w in pool if is_tagged(w)]
        untagged = [to_row(w, "untagged") for w in pool if not is_tagged(w)]
        tagged_pop = round(total_pop * len(tagged) / max(len(pool), 1))
        untagged_pop = total_pop - tagged_pop
        print(f"  pool split: {len(tagged)} tagged / {len(untagged)} untagged")

    # --- clean, dedupe, trim ----------------------------------------------
    final = []
    for bucket, name in ((tagged, "tagged"), (untagged, "untagged")):
        bucket = clean(bucket, drops)
        before = len(bucket)
        bucket = [r for r in bucket if r["openalex_id"] not in already]
        drops["duplicate_from_earlier_batch"] += before - len(bucket)
        if len(bucket) < N_PER_STRATUM:
            print(f"WARNING: only {len(bucket)} usable rows in '{name}' "
                  f"stratum (wanted {N_PER_STRATUM}). Raise POOL_MULTIPLIER "
                  f"or lower N_PER_STRATUM.", file=sys.stderr)
        final.extend(bucket[:N_PER_STRATUM])

    df = pd.DataFrame(final)
    if df.empty:
        sys.exit("No usable rows. Check filters and thresholds.")

    n_tag = int((df["stratum"] == "tagged").sum())
    n_untag = int((df["stratum"] == "untagged").sum())

    # Reweighting factors: sampling is 50/50 by design, so a raw rate computed
    # over this file is NOT a population rate. Multiply each row by its weight
    # (population share / sample share) before reporting any percentage.
    df["population_weight"] = df["stratum"].map({
        "tagged": (tagged_pop / total_pop) / (n_tag / len(df)) if n_tag else 0,
        "untagged": (untagged_pop / total_pop) / (n_untag / len(df)) if n_untag else 0,
    }).round(4)

    # --- metadata sheet ----------------------------------------------------
    meta = pd.DataFrame([
        ("run_timestamp_utc", datetime.now(timezone.utc).isoformat()),
        ("seed", SEED),
        ("fields", ", ".join(str(f) for f in FIELD_IDS)),
        ("year_from", YEAR_FROM),
        ("work_type", WORK_TYPE),
        ("language", LANGUAGE),
        ("oa_filter_applied", "none (abstract presence only)"),
        ("sdg_score_threshold", SDG_SCORE_THRESHOLD),
        ("stratum_split_method", "api_null_filter" if use_null_filter else "local_threshold"),
        ("population_total", total_pop),
        ("population_tagged", tagged_pop),
        ("population_untagged", untagged_pop),
        ("sampled_tagged", n_tag),
        ("sampled_untagged", n_untag),
        ("dropped_abstract_missing", drops["abstract_missing_despite_filter"]),
        ("dropped_abstract_too_short", drops["abstract_too_short"]),
        ("dropped_duplicate_earlier_batch", drops["duplicate_from_earlier_batch"]),
    ], columns=["key", "value"])

    master_cols = ["openalex_id", "title", "year", "abstract", "subfield",
                   "field", "domain", "openalex_sdg_tag",
                   "openalex_sdg_max_score", "stratum", "population_weight",
                   "doi", "oa_status", "cited_by_count"]
    master = df[master_cols]

    # Blind file: no title, no SDG tag, no stratum. Nothing that reveals
    # OpenAlex's own call or lets it be guessed.
    coding = df[["openalex_id", "year", "abstract", "subfield", "field", "domain"]].copy()
    coding["sdg_keyword"] = ""
    coding["contributing_idea"] = ""
    coding["evidence_sentence"] = ""
    coding["relevant_sdg"] = ""
    coding["flagged"] = ""

    master_path = f"{OUT_PREFIX}_master.xlsx"
    coding_path = f"{OUT_PREFIX}_for_coding.xlsx"

    with pd.ExcelWriter(master_path, engine="openpyxl") as xl:
        master.to_excel(xl, sheet_name="works", index=False)
        meta.to_excel(xl, sheet_name="sampling_metadata", index=False)
    coding.to_excel(coding_path, index=False)

    append_processed(df["openalex_id"].tolist())

    print("\n" + "=" * 60)
    print(f"Rows written:          {len(df)} ({n_tag} tagged / {n_untag} untagged)")
    print(f"Dropped, no abstract:  {drops['abstract_missing_despite_filter']}")
    print(f"Dropped, too short:    {drops['abstract_too_short']}")
    print(f"Dropped, seen before:  {drops['duplicate_from_earlier_batch']}")
    print(f"\nMaster:  {master_path}")
    print(f"Coding:  {coding_path}   <- send this one for annotation")
    print("=" * 60)

    # Score distribution, to sanity-check SDG_SCORE_THRESHOLD.
    scores = df["openalex_sdg_max_score"]
    print("\nMax-SDG-score distribution in this sample:")
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        n = int(((scores >= lo) & (scores < lo + 0.2)).sum())
        print(f"  {lo:.1f}-{lo + 0.2:.1f}: {n}")


if __name__ == "__main__":
    main()