"""
Rule-based dialogue policy used by the demo servers.

Until the AudioLLM is trained end to end, the demos answer with canned,
multilingual responses chosen by keyword intent detection. Every turn is
first routed through :class:`~apollo_voice_engine.safety.SafetyClassifier`
so emergencies are escalated to a human regardless of intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from apollo_voice_engine.safety.classifier import SafetyClassifier

DEFAULT_LANGUAGE = "en"

LANGUAGES: list[dict[str, str]] = [
    {"code": "hi", "name": "Hindi", "native": "हिंदी"},
    {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
    {"code": "te", "name": "Telugu", "native": "తెలుగు"},
    {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ"},
    {"code": "en", "name": "English", "native": "English"},
]
LANGUAGE_CODES = frozenset(lang["code"] for lang in LANGUAGES)


class Action(str, Enum):
    RESPOND = "RESPOND"
    TRANSFER_TO_HUMAN = "TRANSFER_TO_HUMAN"


GREETINGS = {
    "hi": "नमस्ते! मैं अपोलो वॉयस असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
    "ta": "வணக்கம்! நான் அப்பல்லோ குரல் உதவியாளர். நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    "te": "నమస్కారం! నేను అపోలో వాయిస్ అసిస్టెంట్. నేను మీకు ఎలా సహాయం చేయగలను?",
    "kn": "ನಮಸ್ಕಾರ! ನಾನು ಅಪೊಲೊ ಧ್ವನಿ ಸಹಾಯಕ. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
    "en": "Hello! I'm Apollo Voice Assistant. How can I help you today?",
}

RESPONSES: dict[str, dict[str, str]] = {
    "appointment": {
        "hi": "आपकी अगली अपॉइंटमेंट कल सुबह 10 बजे डॉ. शर्मा के साथ है। क्या आप इसे पुष्टि करना चाहेंगे?",
        "ta": "உங்கள் அடுத்த சந்திப்பு நாளை காலை 10 மணிக்கு டாக்டர் ஷர்மாவுடன். இதை உறுதிப்படுத்த விரும்புகிறீர்களா?",
        "te": "మీ తదుపరి అపాయింట్‌మెంట్ రేపు ఉదయం 10 గంటలకు డాక్టర్ శర్మతో. మీరు దీన్ని నిర్ధారించాలనుకుంటున్నారా?",
        "kn": "ನಿಮ್ಮ ಮುಂದಿನ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಡಾ. ಶರ್ಮಾ ಅವರೊಂದಿಗೆ. ನೀವು ಇದನ್ನು ದೃಢೀಕರಿಸಲು ಬಯಸುವಿರಾ?",
        "en": "Your next appointment is tomorrow at 10 AM with Dr. Sharma. Would you like to confirm?",
    },
    "pharmacy": {
        "hi": "फार्मेसी सुबह 8 बजे से रात 9 बजे तक खुली है। क्या आपको दवाओं की जानकारी चाहिए?",
        "ta": "மருந்தகம் காலை 8 மணி முதல் இரவு 9 மணி வரை திறந்திருக்கும். மருந்து தகவல் தேவையா?",
        "te": "ఫార్మసీ ఉదయం 8 నుండి రాత్రి 9 వరకు తెరిచి ఉంటుంది. మీకు మందుల సమాచారం కావాలా?",
        "kn": "ಫಾರ್ಮಸಿ ಬೆಳಿಗ್ಗೆ 8 ರಿಂದ ರಾತ್ರಿ 9 ರವರೆಗೆ ತೆರೆದಿರುತ್ತದೆ. ನಿಮಗೆ ಔಷಧ ಮಾಹಿತಿ ಬೇಕೇ?",
        "en": "The pharmacy is open from 8 AM to 9 PM. Do you need medication information?",
    },
    "cardiology": {
        "hi": "कार्डियोलॉजी विभाग तीसरी मंजिल पर है। क्या मुझे आपके लिए अपॉइंटमेंट बुक करें?",
        "ta": "இருதயவியல் துறை மூன்றாவது மாடியில் உள்ளது. உங்களுக்கு சந்திப்பு பதிவு செய்யட்டுமா?",
        "te": "కార్డియాలజీ విభాగం మూడవ అంతస్తులో ఉంది. మీ కోసం అపాయింట్‌మెంట్ బుక్ చేయమంటారా?",
        "kn": "ಹೃದ್ರೋಗ ವಿಭಾಗ ಮೂರನೇ ಮಹಡಿಯಲ್ಲಿದೆ. ನಿಮಗಾಗಿ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಬುಕ್ ಮಾಡಲೇ?",
        "en": "The Cardiology department is on the 3rd floor. Should I book an appointment for you?",
    },
    "greeting": GREETINGS,
    "default": {
        "hi": "मैं समझ गया। क्या मैं इसके बारे में और जानकारी दे सकता हूँ?",
        "ta": "புரிந்துகொண்டேன். இதைப் பற்றி மேலும் தகவல் தர வேண்டுமா?",
        "te": "అర్థమైంది. దీని గురించి మరింత సమాచారం ఇవ్వాలా?",
        "kn": "ಅರ್ಥವಾಯಿತು. ಇದರ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ನೀಡಲೇ?",
        "en": "I understand. Would you like more information about this?",
    },
}

TRANSFER_MESSAGES = {
    "hi": "आपातकाल का पता चला। मैं आपको तुरंत एक स्वास्थ्य विशेषज्ञ से जोड़ रहा हूं। कृपया लाइन पर रहें।",
    "ta": "அவசரநிலை கண்டறியப்பட்டது. நான் உடனடியாக ஒரு சுகாதார நிபுணரை இணைக்கிறேன். தயவுசெய்து காத்திருங்கள்.",
    "te": "అత్యవసర పరిస్థితి గుర్తించబడింది. నేను వెంటనే ఒక ఆరోగ్య నిపుణుడిని కనెక్ట్ చేస్తున్నాను. దయచేసి వేచి ఉండండి.",
    "kn": "ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಪತ್ತೆಯಾಗಿದೆ. ನಾನು ತಕ್ಷಣ ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇನೆ. ದಯವಿಟ್ಟು ನಿರೀಕ್ಷಿಸಿ.",
    "en": "Emergency detected. I'm connecting you with a healthcare professional immediately. Please stay on the line.",
}

# Checked in order; the first intent with a matching keyword wins.
INTENT_KEYWORDS: dict[str, list[str]] = {
    "appointment": ["appointment", "book", "अपॉइंटमेंट", "சந்திப்பு", "అపాయింట్‌మెంట్", "ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್"],
    "pharmacy": [
        "pharmacy",
        "medicine",
        "फार्मेसी",
        "दवा",
        "மருந்தகம்",
        "மருந்து",
        "ఫార్మసీ",
        "మందులు",
        "ಫಾರ್ಮಸಿ",
        "ಔಷಧ",
    ],
    "cardiology": [
        "cardiology",
        "heart",
        "कार्डियोलॉजी",
        "दिल",
        "இருதயவியல்",
        "இருதயம்",
        "కార్డియాలజీ",
        "గుండె",
        "ಹೃದ್ರೋಗ",
        "ಹೃದಯ",
    ],
    "greeting": ["hello", "hi", "hey", "नमस्ते", "வணக்கம்", "నమస్కారం", "ನಮಸ್ಕಾರ"],
}


def _compile(keyword: str) -> re.Pattern[str]:
    # ASCII keywords need word boundaries ("hi" must not match "this").
    if keyword.isascii():
        return re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    return re.compile(re.escape(keyword))


_INTENT_PATTERNS = {
    intent: [_compile(k) for k in keywords] for intent, keywords in INTENT_KEYWORDS.items()
}


def normalize_language(language: str | None) -> str:
    """Return a supported language code, falling back to English."""
    code = (language or "").strip().lower()
    return code if code in LANGUAGE_CODES else DEFAULT_LANGUAGE


def detect_intent(text: str) -> str:
    """Return the first matching intent name, or ``"default"``."""
    for intent, patterns in _INTENT_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            return intent
    return "default"


def _localized(messages: dict[str, str], language: str) -> str:
    return messages.get(language, messages[DEFAULT_LANGUAGE])


def canned_response(text: str, language: str) -> str:
    """Pick the canned response for ``text`` in ``language``."""
    return _localized(RESPONSES[detect_intent(text)], normalize_language(language))


def transfer_message(language: str) -> str:
    return _localized(TRANSFER_MESSAGES, normalize_language(language))


@dataclass
class DialogueTurn:
    """Outcome of one user turn."""

    response: str
    action: Action
    language: str
    safety_level: str
    intent: str
    triggered_keywords: list[str] = field(default_factory=list)

    @property
    def is_transfer(self) -> bool:
        return self.action is Action.TRANSFER_TO_HUMAN


class DialoguePolicy:
    """Safety-first, keyword-based dialogue policy."""

    def __init__(self, classifier: SafetyClassifier | None = None):
        self.classifier = classifier or SafetyClassifier()

    def respond(self, text: str, language: str | None = None) -> DialogueTurn:
        lang = normalize_language(language)
        safety = self.classifier.classify(text)
        if safety.should_transfer:
            return DialogueTurn(
                response=transfer_message(lang),
                action=Action.TRANSFER_TO_HUMAN,
                language=lang,
                safety_level=safety.level.value,
                intent="emergency",
                triggered_keywords=safety.triggered_keywords,
            )
        intent = detect_intent(text)
        return DialogueTurn(
            response=_localized(RESPONSES[intent], lang),
            action=Action.RESPOND,
            language=lang,
            safety_level=safety.level.value,
            intent=intent,
            triggered_keywords=safety.triggered_keywords,
        )
