"""Tests for the rule-based dialogue policy."""

from __future__ import annotations

import pytest

from apollo_voice_engine.dialogue import (
    GREETINGS,
    RESPONSES,
    TRANSFER_MESSAGES,
    Action,
    DialoguePolicy,
    canned_response,
    detect_intent,
    normalize_language,
)


@pytest.mark.parametrize(
    "text,intent",
    [
        ("I need to book an appointment", "appointment"),
        ("नमस्ते, मेरी अपॉइंटमेंट कब है?", "appointment"),
        ("மருந்தகம் எங்கே?", "pharmacy"),
        ("Where is cardiology?", "cardiology"),
        ("Hi there", "greeting"),
        ("ನಮಸ್ಕಾರ", "greeting"),
        ("What is this?", "default"),  # "hi" inside "this" must not match
        ("Which floor?", "default"),
    ],
)
def test_detect_intent(text, intent):
    assert detect_intent(text) == intent


@pytest.mark.parametrize(
    "language,expected", [("hi", "hi"), ("TA", "ta"), (" kn ", "kn"), ("fr", "en"), (None, "en")]
)
def test_normalize_language(language, expected):
    assert normalize_language(language) == expected


def test_every_response_has_all_languages():
    for messages in [GREETINGS, TRANSFER_MESSAGES, *RESPONSES.values()]:
        assert set(messages) == {"hi", "ta", "te", "kn", "en"}


def test_canned_response_localized():
    assert canned_response("pharmacy hours", "te") == RESPONSES["pharmacy"]["te"]
    assert canned_response("pharmacy hours", "zz") == RESPONSES["pharmacy"]["en"]


def test_policy_escalates_emergencies_first():
    turn = DialoguePolicy().respond("I think I'm having a heart attack", "hi")
    assert turn.action is Action.TRANSFER_TO_HUMAN
    assert turn.is_transfer
    assert turn.response == TRANSFER_MESSAGES["hi"]
    assert turn.safety_level == "emergency"
    assert turn.intent == "emergency"
    assert "heart attack" in turn.triggered_keywords


def test_policy_responds_to_intent():
    turn = DialoguePolicy().respond("Where is cardiology? I have a fever", "en")
    assert turn.action is Action.RESPOND
    assert turn.intent == "cardiology"
    assert turn.safety_level == "caution"
    assert turn.response == RESPONSES["cardiology"]["en"]
