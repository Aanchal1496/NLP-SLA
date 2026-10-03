"""Reusable Marathi preprocessing service for legal & financial documents.

Pipeline (all steps preserve Devanagari; see notes below):
1. Unicode normalization (default NFC -- canonical composition of
   consonants + matras; does not fold digits, currency, or case).
2. Optional HTML-tag and URL removal (both OFF by default to preserve info).
3. Safe control-character removal (keeps newline/tab; keeps ZWJ/ZWNJ which
   are significant for Devanagari conjuncts).
4. Whitespace normalization (single newlines become spaces; blank lines stay
   paragraph breaks).
5. Rule-based sentence segmentation (danda, double-danda, ?/!, and "." only
   as a boundary before a likely sentence start; inner dots in abbreviations
   and decimals are masked first).
6. Whitespace-based word tokenization that peels only quote/bracket
   characters, so section numbers, dates, amounts, and case references stay
   intact.
7. Optional stopword removal against a small built-in list (+ custom words).
8. Statistics over the processed output.

Honesty notes (no fake NLP claims):
- NO stemming, lemmatization, or morphological analysis is implemented.
  Inflected forms stay distinct -- verified by tests.
- Sentence segmentation is naive and rule-based; it can mis-split unusual
  abbreviations. Original and processed texts are always returned so callers
  can audit.
- The built-in stopword list is minimal on purpose. Negations are
  deliberately excluded -- dropping them would flip legal meaning.
- No lowercasing is applied (Marathi has no case; lowercasing mixed-script
  Latin could corrupt identifiers such as FIR).
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

VALID_UNICODE_FORMS = ("NFC", "NFKC", "NFD", "NONE")

# Placeholder while protecting inner dots (e.g. in abbreviations/decimals).
_DOT_MASK = ""


def _build_control_pattern() -> "re.Pattern[str]":
    """Match unsafe control chars: C0 (minus tab/newline), DEL and C1."""
    codepoints = (
        list(range(0x00, 0x09))
        + [0x0B, 0x0C]
        + list(range(0x0E, 0x20))
        + list(range(0x7F, 0xA0))
    )
    chars = "".join(chr(c) for c in codepoints)
    return re.compile("[" + re.escape(chars) + "]")


def _build_format_pattern() -> "re.Pattern[str]":
    """Match zero-width/format chars that carry no legal meaning.

    ZWJ (U+200D) and ZWNJ (U+200C) are deliberately NOT in this list --
    they shape Devanagari conjuncts.
    """
    chars = "".join(chr(c) for c in (0x200B, 0x2060, 0xFEFF, 0xFFF9, 0xFFFB))
    return re.compile("[" + re.escape(chars) + "]")


def _build_hspace_pattern() -> "re.Pattern[str]":
    """Match horizontal whitespace runs (space, tab, NBSP, wide spaces)."""
    chars = "".join(
        chr(c)
        for c in (
            0x20, 0x09, 0xA0, 0x1680,
            0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
            0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
            0x202F, 0x205F, 0x3000,
        )
    )
    return re.compile("[" + re.escape(chars) + "]+")


_CONTROL_RE = _build_control_pattern()
_STRIP_FORMAT_RE = _build_format_pattern()
_HSPACE_RUN_RE = _build_hspace_pattern()
_NEWLINE_RUN_RE = re.compile(r"\n{3,}")
_SINGLE_NEWLINE_RE = re.compile(r"(?<!\n)\n(?!\n)")

_URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")
_HTML_TAG_RE = re.compile(r"<[^<>]+>")

DEVANAGARI = "ऀ-ॿ"
# Dot between two word-ish chars (no spaces): abbreviation/decimal innards.
# Commas need no masking -- they never terminate a sentence.
_INNER_DOT_RE = re.compile(
    r"(?<=[0-9A-Za-z" + DEVANAGARI + r"])\.(?=[0-9A-Za-z" + DEVANAGARI + r"])"
)
# Tail of an abbreviation run (e.g. "बी.सी." with masked inner dots):
# a dot-boundary right after such a tail is skipped, not a sentence split.
# The "+" lets a unit hold matras/digits ("बी" = consonant + vowel sign).
_ABBR_TAIL_RE = re.compile(
    r"(?:[0-9A-Za-z" + DEVANAGARI + r"]+(?:\.|" + _DOT_MASK + r")){2,}$"
)
# Sentence terminators: dandas, ?/!, or '.' before a plausible sentence start.
_SENT_END_RE = re.compile(
    r"[।॥?!]+|\.(?=\s*(?:[" + DEVANAGARI + r"A-Z\"'‘“(\[]|$))"
)

_WS_SPLIT_RE = re.compile(r"\s+")
# Only quotes/brackets are peeled from token edges -- never digits, currency,
# slashes, dots, commas, or dandas.
_TOKEN_EDGE_CHARS = "\"'‘’“”()[]{}<>«»…„‟‹›"

#: Minimal built-in Marathi stopwords. Negations excluded by design.
MARATHI_STOPWORDS: frozenset[str] = frozenset(
    {
        "आहे", "आहेत", "होता", "होती", "होते",
        "असा", "असे", "अशी",
        "आणि", "किंवा", "पण", "परंतु", "मात्र", "की",
        "कारण", "म्हणून", "साठी",
        "ने", "ला", "ना", "चा", "ची", "चे", "च्या", "चं",
        "हा", "ही", "हे", "तो", "ती", "ते", "या", "त्या",
        "जो", "जी", "जे",
        "काय", "कसा", "कसे", "कशी", "कुठे", "केव्हा",
        "जेव्हा", "तेव्हा", "जसे", "तसे",
        "खूप", "फार", "देखील", "सुद्धा",
        "तर", "जर", "मग", "आता", "आज", "इथे", "तिथे",
        "सर्व", "प्रत्येक", "काही", "कोण",
        "मी", "आम्ही", "तू", "तुम्ही",
        "त्याने", "त्यांनी", "तिने", "तिला", "त्याला",
        "होय",
    }
)


@dataclass
class PreprocessOptions:
    """Knobs for :func:`preprocess_marathi_text` (all optional)."""

    unicode_form: str = "NFC"  # NFC | NFKC | NFD | NONE
    remove_urls: bool = False
    remove_html: bool = False
    remove_stopwords: bool = False
    custom_stopwords: list[str] | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.unicode_form not in VALID_UNICODE_FORMS:
            raise ValueError(
                f"unicode_form must be one of {VALID_UNICODE_FORMS}, "
                f"got {self.unicode_form!r}."
            )


@dataclass
class TextStatistics:
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    vocab_size: int = 0


@dataclass
class PreprocessResult:
    original_text: str = ""
    processed_text: str = ""
    sentences: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    stats: TextStatistics = field(default_factory=TextStatistics)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Apply a Unicode normalization form; ``NONE`` returns text unchanged."""
    if form not in VALID_UNICODE_FORMS:
        raise ValueError(f"unicode_form must be one of {VALID_UNICODE_FORMS}.")
    if form == "NONE":
        return text
    return unicodedata.normalize(form, text)


def _strip_unsafe_chars(text: str) -> str:
    text = _CONTROL_RE.sub("", text)
    return _STRIP_FORMAT_RE.sub("", text)


def clean_text(text: str, options: PreprocessOptions) -> str:
    """Normalize + clean raw text; legal/financial identifiers untouched."""
    cleaned = normalize_unicode(text, options.unicode_form)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if options.remove_html:
        cleaned = html.unescape(cleaned)
        cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    if options.remove_urls:
        cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _strip_unsafe_chars(cleaned)
    # Per-line edge whitespace (indentation); blank lines survive as "".
    lines = [_HSPACE_RUN_RE.sub(" ", ln).strip(" ") for ln in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    # Single newlines are line-wraps -> space; blank lines stay boundaries.
    cleaned = _SINGLE_NEWLINE_RE.sub(" ", cleaned)
    cleaned = _HSPACE_RUN_RE.sub(" ", cleaned)
    cleaned = _NEWLINE_RUN_RE.sub("\n\n", cleaned)
    return cleaned.strip(" \n")


def segment_sentences(text: str) -> list[str]:
    """Naive rule-based segmentation; keeps terminators attached."""
    if not text or not text.strip():
        return []
    sentences: list[str] = []
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue
        masked = _INNER_DOT_RE.sub(_DOT_MASK, block)
        start = 0
        for match in _SENT_END_RE.finditer(masked):
            if match.group().startswith(".") and _ABBR_TAIL_RE.search(
                masked[start : match.end()]
            ):
                continue  # trailing dot of an abbreviation run, not a boundary
            piece = masked[start : match.end()].strip()
            if piece:
                sentences.append(piece.replace(_DOT_MASK, "."))
            start = match.end()
        tail = masked[start:].strip()
        if tail:
            sentences.append(tail.replace(_DOT_MASK, "."))
    return sentences


def tokenize_words(text: str) -> list[str]:
    """Whitespace tokens with only quote/bracket edges peeled."""
    if not text or not text.strip():
        return []
    tokens: list[str] = []
    for piece in _WS_SPLIT_RE.split(text.strip()):
        token = piece.strip(_TOKEN_EDGE_CHARS)
        if token:
            tokens.append(token)
    return tokens


def remove_stopwords(
    tokens: list[str], custom: list[str] | tuple[str, ...] | None = None
) -> list[str]:
    """Drop minimal built-in (+ custom) Marathi stopwords; order kept."""
    stopwords = set(MARATHI_STOPWORDS)
    if custom:
        stopwords.update(custom)
    return [tok for tok in tokens if tok not in stopwords]


def compute_stats(tokens: list[str], sentences: list[str], text: str) -> TextStatistics:
    return TextStatistics(
        char_count=len(text),
        word_count=len(tokens),
        sentence_count=len(sentences),
        vocab_size=len(set(tokens)),
    )


def preprocess_marathi_text(
    text: str, options: PreprocessOptions | None = None
) -> PreprocessResult:
    """Full pipeline. Empty/whitespace input -> safe empty result (no crash)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    opts = options or PreprocessOptions()
    if not text.strip():
        return PreprocessResult(original_text=text)
    processed = clean_text(text, opts)
    if not processed:
        return PreprocessResult(original_text=text)
    sentences = segment_sentences(processed)
    tokens = tokenize_words(processed)
    if opts.remove_stopwords:
        tokens = remove_stopwords(tokens, opts.custom_stopwords)
    return PreprocessResult(
        original_text=text,
        processed_text=processed,
        sentences=sentences,
        tokens=tokens,
        stats=compute_stats(tokens, sentences, processed),
    )
