# -*- coding: utf-8 -*-
"""Tests for the hybrid baseline NER (regex + gazetteer).

No CCE-2 guidelines or gold entity spans exist in this workspace, so the
labels asserted here are the documented BASELINE labels, and the
entity_prf test uses toy spans only -- no project-level scores reported.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.services import ner as ner_svc
from app.services.ner import (
    BASELINE_LABELS,
    EntityMention,
    NerOptions,
    entity_prf,
    extract_entities,
)

client = TestClient(app)

PARA = (
    "मुंबई उच्च न्यायालयाने दिनांक १२/०५/२०२४ रोजी कलम ३०२ अंतर्गत "
    "सी.आर. नं. १२३/२०२४ मधील आरोपीस ₹५,००,००० दंड ठोठावला।"
)


def _labels(result):
    return [(m.text, m.label) for m in result.entities]


def test_baseline_labels_documented():
    assert set(BASELINE_LABELS) == {
        "DATE", "MONEY", "CASE_NUMBER", "SECTION", "ORGANIZATION", "PERSON",
    }


def test_date_devanagari_and_ascii():
    assert ("१२/०५/२०२४", "DATE") in _labels(extract_entities("दिनांक १२/०५/२०२४ रोजी"))
    assert ("12/05/2024", "DATE") in _labels(extract_entities("on 12/05/2024 filed"))


def test_money_symbols_and_words():
    assert ("₹५,००,०००", "MONEY") in _labels(extract_entities("रक्कम ₹५,००,००० देय"))
    res = extract_entities("दंड ५००० रुपये ठोठावला")
    assert any(label == "MONEY" and "५०००" in text for text, label in _labels(res))


def test_section_reference():
    assert ("कलम ३०२", "SECTION") in _labels(extract_entities("कलम ३०२ अंतर्गत गुन्हा"))


def test_case_number_with_prefix():
    res = extract_entities("सी.आर. नं. १२३/२०२४ अन्वये तपास")
    assert ("सी.आर. नं. १२३/२०२४", "CASE_NUMBER") in _labels(res)


def test_organization_gazetteer_with_inflection():
    res = extract_entities("मुंबई उच्च न्यायालयाने निर्णय दिला।")
    assert ("मुंबई उच्च न्यायालय", "ORGANIZATION") in _labels(res)


def test_person_via_custom_gazetteer():
    opts = NerOptions(custom_gazetteer={"राहुल शर्मा": "PERSON"})
    res = extract_entities("फिर्यादी राहुल शर्मा हजर झाला।", opts)
    assert ("राहुल शर्मा", "PERSON") in _labels(res)
    assert res.entities[0].method == "gazetteer"


def test_longer_nested_phrase_wins():
    res = extract_entities("भारतीय रिझर्व्ह बँकेने परिपत्रक काढले।")
    orgs = [m for m in res.entities if m.label == "ORGANIZATION"]
    assert len(orgs) == 1
    assert orgs[0].text == "भारतीय रिझर्व्ह बँक"


def test_date_beats_case_number_on_true_date():
    res = extract_entities("दिनांक १२/०५/२०२४ रोजी")
    assert len(res.entities) == 1
    assert res.entities[0].label == "DATE"
    assert res.entities[0].text == "१२/०५/२०२४"


def test_repeated_entities_have_distinct_offsets():
    res = extract_entities("₹५०० भरले आणि पुन्हा ₹५०० भरले।")
    money = [m for m in res.entities if m.label == "MONEY"]
    assert len(money) == 2
    assert money[0].start != money[1].start
    assert money[0].text == money[1].text == "₹५००"


def test_offset_invariant_on_paragraph():
    res = extract_entities(PARA)
    assert len(res.entities) >= 5
    for m in res.entities:
        assert PARA[m.start : m.end] == m.text
        assert m.method in ("regex", "gazetteer")


def test_no_duplicate_spans():
    res = extract_entities(PARA)
    keys = [(m.start, m.end, m.label) for m in res.entities]
    assert len(keys) == len(set(keys))


def test_generic_word_alone_is_not_entity():
    res = extract_entities("न्यायालयात आज गर्दी होती।")
    assert all(m.text != "न्यायालय" for m in res.entities)
    assert all(m.label != "ORGANIZATION" for m in res.entities)


def test_entities_sorted_by_start():
    res = extract_entities(PARA)
    starts = [m.start for m in res.entities]
    assert starts == sorted(starts)


def test_options_disable_groups():
    no_regex = extract_entities(PARA, NerOptions(regex_enabled=False))
    assert no_regex.entities
    assert all(m.method == "gazetteer" for m in no_regex.entities)
    no_gaz = extract_entities(PARA, NerOptions(gazetteer_enabled=False))
    assert no_gaz.entities
    assert all(m.method == "regex" for m in no_gaz.entities)


def test_empty_and_invalid_input():
    assert extract_entities("").entities == []
    assert extract_entities("   ").entities == []
    try:
        extract_entities(None)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_entity_prf_toy_only():
    gold = [
        EntityMention(text="a", label="DATE", start=0, end=1, method="regex"),
        EntityMention(text="b", label="MONEY", start=2, end=3, method="regex"),
    ]
    perfect = entity_prf(list(gold), list(gold))
    assert perfect == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    partial = entity_prf(
        [gold[0], EntityMention(text="x", label="DATE", start=9, end=10, method="regex")],
        list(gold),
    )
    assert partial == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    empty = entity_prf([], list(gold))
    assert empty == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_api_entities_success_and_offsets():
    resp = client.post("/api/v1/predict/entities", json={"text": PARA})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == len(body["entities"])
    assert body["model"]["type"] in ("hybrid_baseline", "hybrid_baseline+indicbert_ner")
    for ent in body["entities"]:
        assert PARA[ent["start"] : ent["end"]] == ent["text"]
        assert set(ent) == {"text", "label", "start", "end", "method"}


def test_api_entities_empty_rejected():
    resp = client.post("/api/v1/predict/entities", json={"text": "   "})
    assert resp.status_code == 422
