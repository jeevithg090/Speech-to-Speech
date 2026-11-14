"""Tests for the multilingual safety classifier."""

from __future__ import annotations

import pytest

from apollo_voice_engine.safety import SafetyClassifier, SafetyLevel, check_safety


@pytest.fixture(scope="module")
def classifier() -> SafetyClassifier:
    return SafetyClassifier()


@pytest.mark.parametrize(
    "text,keyword",
    [
        ("I need an ambulance now!", "ambulance"),
        ("He is having a HEART ATTACK", "heart attack"),
        ("मुझे छाती में दर्द हो रहा है", "छाती में दर्द"),
        ("நெஞ்சு வலி", "நெஞ்சு வலி"),
        ("ఛాతీ నొప్పి గా ఉంది", "ఛాతీ నొప్పి"),
        ("ಎದೆ ನೋವು", "ಎದೆ ನೋವು"),
    ],
)
def test_emergency_in_each_language(classifier, text, keyword):
    result = classifier.classify(text)
    assert result.level is SafetyLevel.EMERGENCY
    assert result.should_transfer
    assert keyword in result.triggered_keywords


@pytest.mark.parametrize("text", ["I have a fever", "मुझे बुखार है", "தலைச்சுற்று"])
def test_caution(classifier, text):
    result = classifier.classify(text)
    assert result.level is SafetyLevel.CAUTION
    assert not result.should_transfer


@pytest.mark.parametrize(
    "text",
    [
        "What time does the pharmacy open?",
        "This is ridiculous",  # contains "icu"
        "I just got back from Spain",  # contains "pain"
        "నా అపాయింట్‌మెంట్ ఏమిటి?",
        "",
    ],
)
def test_safe_without_substring_false_positives(classifier, text):
    result = classifier.classify(text)
    assert result.level is SafetyLevel.SAFE
    assert result.triggered_keywords == []


def test_emergency_takes_precedence_over_caution(classifier):
    result = classifier.classify("chest pain and fever")
    assert result.level is SafetyLevel.EMERGENCY
    assert "chest pain" in result.triggered_keywords


def test_triggered_keywords_are_deterministic(classifier):
    first = classifier.classify("emergency, ambulance, bleeding").triggered_keywords
    second = classifier.classify("emergency, ambulance, bleeding").triggered_keywords
    assert first == second == sorted(first)


@pytest.mark.parametrize(
    "response",
    [
        "Take two tablets of paracetamol every 6 hours",
        "The dosage is 500 mg",
        "I would diagnose this as flu",
        "You need a prescription",
    ],
)
def test_unsafe_responses_flagged(classifier, response):
    is_safe, _ = classifier.check_response_safety(response)
    assert not is_safe


@pytest.mark.parametrize(
    "response",
    ["The pharmacy is open from 8 AM to 9 PM.", "Upload the image to imgur", "Amgen is a company"],
)
def test_safe_responses_pass(classifier, response):
    assert classifier.check_response_safety(response)[0]


def test_transfer_message_fallback(classifier):
    assert "healthcare professional" in classifier.get_transfer_message("xx")
    assert classifier.get_transfer_message("hi") != classifier.get_transfer_message("en")


def test_check_safety_helper():
    assert check_safety("call an ambulance").should_transfer
