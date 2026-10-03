# -*- coding: utf-8 -*-
"""Tests for the reusable Marathi preprocessing service + /preprocess route."""

import unicodedata

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import marathi_preprocess as mp
from app.services.marathi_preprocess import PreprocessOptions

client = TestClient(app)

SENT1 = "मुंबई उच्च न्यायालयाने मालमत्ता वादात महत्त्वाचा निर्णय दिला।"
SENT2 = "करारानुसार देय रक्कम ₹५,००,००० इतकी निश्चित करण्यात आली।"
LEGAL = "कलम ३०२ अंतर्गत दिनांक १२/०५/२०२४ रोजी ₹५,००,००० देय आहेत।"


# --- Unicode normalization -------------------------------------------------

def test_devanagari_intact_full_pipeline():
    res = mp.preprocess_marathi_text(SENT1 + " " + SENT2)
    assert "न्यायालयाने" in res.processed_text
    assert "₹५,००,०००" in res.processed_text
    assert res.original_text == SENT1 + " " + SENT2


def test_nfd_input_composes_to_nfc():
    # Most Devanagari text is already NFC-stable, so exercise composition
    # with a decomposable character alongside Marathi content.
    composed = "क्रमांक café १२३"
    nfd = unicodedata.normalize("NFD", composed)
    assert nfd != composed  # sanity: decomposition actually differs
    res = mp.preprocess_marathi_text(nfd)
    assert res.processed_text == unicodedata.normalize("NFC", composed)
    assert "café" in res.processed_text
    assert "क्रमांक" in res.tokens


def test_unicode_form_none_passes_through():
    nfd = unicodedata.normalize("NFD", "मुंबई")
    assert mp.normalize_unicode(nfd, "NONE") == nfd


def test_invalid_unicode_form_raises():
    with pytest.raises(ValueError):
        PreprocessOptions(unicode_form="XYZ")
    with pytest.raises(ValueError):
        mp.normalize_unicode("abc", "XYZ")


# --- Whitespace / control characters ---------------------------------------

def test_whitespace_normalized_paragraphs_kept():
    res = mp.preprocess_marathi_text("  खूप   जागा\t\tआहे\n\n\n\nपुढील   परा ")
    assert "खूप जागा आहे" in res.processed_text
    assert "\n\n" in res.processed_text
    assert "\n\n\n" not in res.processed_text


def test_control_chars_removed_zwj_kept():
    zwj = chr(0x200D)
    text = "मजकूर" + chr(0) + chr(7) + "पुढे " + "अ" + zwj + "ब"
    res = mp.preprocess_marathi_text(text)
    assert chr(0) not in res.processed_text
    assert chr(7) not in res.processed_text
    assert zwj in res.processed_text
    assert "मजकूरपुढे" in res.processed_text


# --- Optional URL / HTML removal --------------------------------------------

def test_urls_kept_by_default_removed_when_asked():
    text = "पहा https://example.com/karar अधिक माहिती"
    assert "https://example.com/karar" in mp.preprocess_marathi_text(text).processed_text
    res = mp.preprocess_marathi_text(text, PreprocessOptions(remove_urls=True))
    assert "http" not in res.processed_text
    assert "example.com" not in res.processed_text
    assert "पहा" in res.processed_text and "अधिक" in res.tokens


def test_html_kept_by_default_removed_when_asked():
    text = "<p>निर्णय</p> <b>अंतिम</b> आहे"
    assert "<p>" in mp.preprocess_marathi_text(text).processed_text
    res = mp.preprocess_marathi_text(text, PreprocessOptions(remove_html=True))
    assert "<p>" not in res.processed_text
    assert "निर्णय" in res.processed_text and "अंतिम" in res.tokens


# --- Segmentation / tokenization ---------------------------------------------

def test_sentence_segmentation_danda():
    res = mp.preprocess_marathi_text(SENT1 + " " + SENT2)
    assert len(res.sentences) == 2
    assert res.sentences[0].endswith("।")
    assert "निर्णय दिला।" in res.sentences[0]


def test_no_split_inside_abbreviation_or_decimal():
    res = mp.preprocess_marathi_text("बी.सी. कार्यालयात अर्ज केला। रक्कम १२.५ लाख आहे।")
    assert len(res.sentences) == 2
    assert "बी.सी." in res.sentences[0]
    assert "१२.५" in res.sentences[1]


def test_legal_identifiers_preserved_in_tokens():
    res = mp.preprocess_marathi_text(LEGAL)
    assert "कलम" in res.tokens
    assert "३०२" in res.tokens
    assert "१२/०५/२०२४" in res.tokens
    assert "₹५,००,०००" in res.tokens
    assert "।" not in res.tokens  # danda stays attached, never a lone token


def test_mixed_script_tokens():
    res = mp.preprocess_marathi_text("FIR No. 123/2024 दाखल झाला।")
    assert "FIR" in res.tokens
    assert "No." in res.tokens
    assert "123/2024" in res.tokens
    assert "दाखल" in res.tokens
    assert len(res.sentences) == 1  # 'No.' must not split the sentence


def test_case_reference_survives_pipeline():
    text = "सी.आर. नं. १२३/२०२४ अन्वये तपास सुरू आहे।"
    res = mp.preprocess_marathi_text(text)
    assert "१२३/२०२४" in res.processed_text
    assert "१२३/२०२४" in res.tokens


# --- Stopwords ----------------------------------------------------------------

def test_stopwords_off_by_default_on_when_asked():
    text = "करार आणि हक्क कायम आहे"
    assert "आणि" in mp.preprocess_marathi_text(text).tokens
    res = mp.preprocess_marathi_text(text, PreprocessOptions(remove_stopwords=True))
    assert "आणि" not in res.tokens
    assert "आहे" not in res.tokens
    assert "करार" in res.tokens and "हक्क" in res.tokens


def test_negation_never_removed():
    res = mp.preprocess_marathi_text(
        "करार रद्द नाही झाला", PreprocessOptions(remove_stopwords=True)
    )
    assert "नाही" in res.tokens


def test_custom_stopwords_extend_builtin():
    res = mp.preprocess_marathi_text(
        "करार रद्द झाला", PreprocessOptions(remove_stopwords=True, custom_stopwords=["करार"])
    )
    assert "करार" not in res.tokens
    assert "रद्द" in res.tokens


def test_no_stemming_inflections_stay_distinct():
    res = mp.preprocess_marathi_text("काम केली। काम केले।")
    assert "केली।" in res.tokens
    assert "केले।" in res.tokens


# --- Statistics / result shape --------------------------------------------------

def test_stats_correct_for_known_example():
    res = mp.preprocess_marathi_text("राम घरी गेला।")
    assert res.sentences == ["राम घरी गेला।"]
    assert res.tokens == ["राम", "घरी", "गेला।"]
    assert res.stats.sentence_count == 1
    assert res.stats.word_count == 3
    assert res.stats.vocab_size == 3
    assert res.stats.char_count == len(res.processed_text)


def test_numbers_and_punctuation_not_blindly_removed():
    res = mp.preprocess_marathi_text("कलम ४२०, दिनांक ०१-०२-२०२३।")
    assert "४२०" in res.processed_text
    # Trailing punctuation stays attached to its token (never stripped blindly).
    assert "४२०," in res.tokens
    assert "०१-०२-२०२३।" in res.tokens


# --- Empty / invalid input -------------------------------------------------------

def test_empty_and_whitespace_input_safe():
    for bad in ("", "   ", "\n\n  \n"):
        res = mp.preprocess_marathi_text(bad)
        assert res.processed_text == ""
        assert res.tokens == []
        assert res.sentences == []
        assert res.stats.word_count == 0
        assert res.original_text == bad


def test_non_string_raises_type_error():
    with pytest.raises(TypeError):
        mp.preprocess_marathi_text(None)
    with pytest.raises(TypeError):
        mp.preprocess_marathi_text(123)


# --- API route ---------------------------------------------------------------------

def test_api_preprocess_success_marathi():
    resp = client.post("/api/v1/preprocess", json={"text": SENT1 + " " + SENT2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["original_text"] == SENT1 + " " + SENT2
    assert "₹५,००,०००" in body["processed_text"]
    assert len(body["sentences"]) == 2
    assert body["stats"]["word_count"] == len(body["tokens"])
    assert body["stats"]["sentence_count"] == len(body["sentences"])
    assert body["stats"]["char_count"] == len(body["processed_text"])


def test_api_preprocess_empty_string_ok():
    resp = client.post("/api/v1/preprocess", json={"text": ""})
    assert resp.status_code == 200, resp.text
    assert resp.json()["tokens"] == []


def test_api_preprocess_with_options():
    resp = client.post(
        "/api/v1/preprocess",
        json={"text": "करार आणि हक्क", "options": {"remove_stopwords": True}},
    )
    assert resp.status_code == 200, resp.text
    assert "आणि" not in resp.json()["tokens"]


def test_api_preprocess_bad_option_rejected():
    resp = client.post(
        "/api/v1/preprocess", json={"text": "चाचणी", "options": {"unicode_form": "XYZ"}}
    )
    assert resp.status_code == 422
