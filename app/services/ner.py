"""Hybrid baseline NER for Marathi legal & financial text.

SCHEMA STATUS (verified 2026-10-02): no CCE-2 annotation guidelines, no
entity-annotated dataset (spans / BIO / columns / plain), and no trained or
loadable NER model exist in this workspace (``models/`` is empty; the only
installed spaCy pipeline is English-only ``en_core_web_sm``). There is
therefore NO established annotation scheme to reuse. The label set below is
**baseline-defined** and must be reconciled with the CCE-2 guidelines when
they are supplied -- see ``BASELINE_LABELS``.

IMPLEMENTATION TYPE: transparent hybrid baseline -- NOT a trained NER model.
- ``regex``: deterministic patterns for DATE, MONEY, CASE_NUMBER, SECTION.
- ``gazetteer``: literal matching of curated multi-word legal/financial
  proper names (ORGANIZATION plus caller-supplied PERSON/others). The
  built-in list is illustrative, NOT dataset-derived (no dataset exists);
  pass ``custom_gazetteer`` for project terms.
- No dataset entity vocabulary is used (none can be reliably extracted).

Rules:
- Matching runs on the ORIGINAL text, so ``text[start:end] == entity.text``
  holds by construction (tested as an invariant).
- Overlaps resolve deterministically: candidates sort by
  ``(start, -length, label_priority)`` and are accepted greedily when they
  do not overlap an accepted span. Priority: DATE > MONEY > CASE_NUMBER >
  SECTION > ORGANIZATION > PERSON > custom labels (alphabetical).
- Identical spans deduplicate to one mention.
- The gazetteer holds multi-word proper names only -- no generic singletons
  (e.g. bare "न्यायालय" is never an entity), so generic words are not
  labelled on every occurrence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BASELINE_LABELS = ("DATE", "MONEY", "CASE_NUMBER", "SECTION", "ORGANIZATION", "PERSON")

_LABEL_PRIORITY = {label: i for i, label in enumerate(BASELINE_LABELS)}

METHOD_REGEX = "regex"
METHOD_GAZETTEER = "gazetteer"

_DIG = "0-9०-९"  # ASCII + Devanagari digits

DATE_PATTERNS = (
    # 12/05/2024 and १२/०५/२०२४
    r"[" + _DIG + r"]{1,2}/[" + _DIG + r"]{1,2}/[" + _DIG + r"]{4}",
    # 12-05-2024 and १२-०५-२०२४
    r"[" + _DIG + r"]{1,2}-[" + _DIG + r"]{1,2}-[" + _DIG + r"]{4}",
)

MONEY_PATTERNS = (
    # ₹5,00,000 / ₹ ५,००,०००
    r"₹\s?[" + _DIG + r"][" + _DIG + r",.]*",
    # 500000 रुपये / ५,००,००० रू.
    r"[" + _DIG + r"][" + _DIG + r",.]*(?:\s?(?:रुपये|रू\.?))",
)

CASE_NUMBER_PATTERNS = (
    # सी.आर. नं. १२३/२०२४, CR No. 123/2024, No. 123/2024
    r"(?:[A-Za-zऀ-ॿ]+\.\s*)*(?:No\.?|नं\.?|नंबर|क्रमांक)\s?[" + _DIG + r"]+/[" + _DIG + r"]+",
    # bare 123/2024, १२३/२०२४ (DATE wins true-date overlaps by priority)
    r"[" + _DIG + r"]{1,6}/[" + _DIG + r"]{2,4}",
)

SECTION_PATTERNS = (
    # कलम ३०२, धारा 420, Section 302, कलम ३०२(अ)
    r"(?:कलम|धारा|[Ss]ection)\s?[" + _DIG + r"]+(?:\s?\([0-9A-Za-zऀ-ॿ]+\))?",
)

_REGEX_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DATE", DATE_PATTERNS),
    ("MONEY", MONEY_PATTERNS),
    ("CASE_NUMBER", CASE_NUMBER_PATTERNS),
    ("SECTION", SECTION_PATTERNS),
)

#: Illustrative built-in gazetteer {phrase: label}. NOT dataset-derived.
DEFAULT_GAZETTEER: dict[str, str] = {
    "मुंबई उच्च न्यायालय": "ORGANIZATION",
    "सर्वोच्च न्यायालय": "ORGANIZATION",
    "महाराष्ट्र शासन": "ORGANIZATION",
    "भारतीय रिझर्व्ह बँक": "ORGANIZATION",
    "रिझर्व्ह बँक": "ORGANIZATION",
    "भारतीय दंड संहिता": "ORGANIZATION",
    "मोटार वाहन कायदा": "ORGANIZATION",
}

def _continues_word(char: str) -> bool:
    """True when `char` would continue a word past a phrase edge.

    Marathi inflects by suffixation (न्यायालय + ाने), so trailing vowel
    signs and marks (matras, anusvara) do NOT block a gazetteer match --
    only letters, digits, and underscore do. The entity span stays the
    base phrase; offsets remain exact.
    """
    if "A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9":
        return True
    if char == "_":
        return True
    if "अ" <= char <= "ह" or "०" <= char <= "९":
        return True
    return False


def _is_word_char(char: str) -> bool:
    return _continues_word(char)


@dataclass
class NerOptions:
    """Knobs for :func:`extract_entities` (all optional)."""

    regex_enabled: bool = True
    gazetteer_enabled: bool = True
    custom_gazetteer: dict[str, str] | None = None

    def active_gazetteer(self) -> dict[str, str]:
        merged = dict(DEFAULT_GAZETTEER)
        if self.custom_gazetteer:
            merged.update(self.custom_gazetteer)
        return merged


@dataclass(frozen=True)
class EntityMention:
    text: str
    label: str
    start: int
    end: int
    method: str


@dataclass
class NerResult:
    original_text: str
    entities: list[EntityMention] = field(default_factory=list)


def _label_rank(label: str) -> int:
    return _LABEL_PRIORITY.get(label, len(_LABEL_PRIORITY))


def _boundary_ok(text: str, start: int, end: int) -> bool:
    if start > 0 and _is_word_char(text[start - 1]):
        return False
    if end < len(text) and _is_word_char(text[end]):
        return False
    return True


def _regex_candidates(text: str) -> list[EntityMention]:
    found: list[EntityMention] = []
    for label, patterns in _REGEX_RULES:
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.start(), match.end()
                snippet = text[start:end].strip(" \t")
                if not snippet:
                    continue
                offset = match.group().find(snippet)
                start += offset
                end = start + len(snippet)
                found.append(
                    EntityMention(
                        text=snippet, label=label, start=start, end=end,
                        method=METHOD_REGEX,
                    )
                )
    return found


def _gazetteer_candidates(
    text: str, gazetteer: dict[str, str]
) -> list[EntityMention]:
    found: list[EntityMention] = []
    for phrase, label in gazetteer.items():
        if not phrase:
            continue
        search_from = 0
        while True:
            at = text.find(phrase, search_from)
            if at < 0:
                break
            end = at + len(phrase)
            if _boundary_ok(text, at, end):
                found.append(
                    EntityMention(
                        text=phrase, label=label, start=at, end=end,
                        method=METHOD_GAZETTEER,
                    )
                )
            search_from = at + max(len(phrase), 1)
    return found


def _resolve(candidates: list[EntityMention]) -> list[EntityMention]:
    """Deterministic overlap resolution + identical-span dedup."""
    seen: set[tuple[int, int, str]] = set()
    unique: list[EntityMention] = []
    for cand in candidates:
        key = (cand.start, cand.end, cand.label)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    unique.sort(key=lambda m: (m.start, -(m.end - m.start), _label_rank(m.label), m.label))
    accepted: list[EntityMention] = []
    for cand in unique:
        if all(cand.end <= other.start or cand.start >= other.end for other in accepted):
            accepted.append(cand)
    return accepted


def extract_entities(
    text: str, options: NerOptions | None = None
) -> NerResult:
    """Extract baseline entities from the ORIGINAL text (offsets align)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    if not text.strip():
        return NerResult(original_text=text, entities=[])
    opts = options or NerOptions()
    candidates: list[EntityMention] = []
    if opts.regex_enabled:
        candidates.extend(_regex_candidates(text))
    if opts.gazetteer_enabled:
        candidates.extend(_gazetteer_candidates(text, opts.active_gazetteer()))
    return NerResult(original_text=text, entities=_resolve(candidates))


def entity_prf(
    predicted: list[EntityMention], gold: list[EntityMention]
) -> dict[str, float]:
    """Exact span+label micro precision/recall/F1 (utility for future gold data).

    No annotated spans exist in this project yet, so this is exercised only
    on toy inputs in tests -- no project-level scores are reported.
    """
    pred_keys = {(m.start, m.end, m.label) for m in predicted}
    gold_keys = {(m.start, m.end, m.label) for m in gold}
    correct = len(pred_keys & gold_keys)
    precision = correct / len(pred_keys) if pred_keys else 0.0
    recall = correct / len(gold_keys) if gold_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
