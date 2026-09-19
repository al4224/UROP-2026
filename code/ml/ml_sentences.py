"""Sentence splitting and evidence matching for scientific abstracts."""

import re
import unicodedata

GREEK = {"\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta",
         "\u03bc": "mu", "\u03c0": "pi", "\u03c3": "sigma", "\u03c9": "omega",
         "\u03bb": "lambda", "\u03b8": "theta", "\u03ba": "kappa", "\u03b5": "epsilon"}
# Full stops that end these do not end a sentence.
ABBREVIATIONS = ("fig", "figs", "eq", "eqs", "ref", "refs", "no", "nos", "vs", "cf",
                 "approx", "et al", "e.g", "i.e", "dr", "prof", "st", "ca", "vol",
                 "min", "max", "wt", "mol", "aq", "tab", "resp", "etc")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\u201c])")
INITIAL = re.compile(r"\b[A-Z]\.$")


def split_sentences(text):
    """
    Split on sentence punctuation, but not inside brackets, not after a known
    abbreviation, and not after an initial such as "Dugas and J. Irwin".
    """
    masked = list(text)
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch in ".!?" and depth:
            masked[i] = "\x00"
    blanked = "".join(masked)
    for abbr in ABBREVIATIONS:
        for match in re.finditer(rf"\b{re.escape(abbr)}\.", blanked, re.IGNORECASE):
            masked[match.end() - 1] = "\x00"
    blanked = "".join(masked)
    for match in re.finditer(r"\b[A-Z]\.", blanked):
        masked[match.end() - 1] = "\x00"

    sentences, cursor = [], 0
    for piece in SENTENCE_BREAK.split("".join(masked)):
        sentences.append(text[cursor:cursor + len(piece)].strip())
        cursor += len(piece) + 1                  # the split consumed one space
    return [s for s in sentences if s]


def squash(text):
    """Strip everything that differs between a quote and its source."""
    text = unicodedata.normalize("NFKD", "".join(GREEK.get(c, c) for c in text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def find_evidence(sentences, evidence):
    """
    Index of the sentence holding `evidence`, or None. Quotes are matched after
    squashing because coders retype dashes, Greek letters and spacing.
    """
    target = squash(evidence)
    if not target:
        return None
    for i, sentence in enumerate(sentences):
        if target in squash(sentence) or squash(sentence) in target:
            return i
    best, score = None, 0.0
    for i, sentence in enumerate(sentences):      # edited quotes: longest overlap
        shared = _overlap(squash(sentence), target)
        if shared > score:
            best, score = i, shared
    return best if score >= 0.6 else None


def _overlap(a, b):
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    window = len(shorter)
    best = max((sum(x == y for x, y in zip(longer[i:i + window], shorter))
                for i in range(0, max(1, len(longer) - window + 1))), default=0)
    return best / window
