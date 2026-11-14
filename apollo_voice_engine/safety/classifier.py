"""
Apollo Voice Engine - Safety Classifier

Detects emergency keywords and unsafe medical situations
for immediate escalation to human operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety classification levels."""

    SAFE = "safe"
    CAUTION = "caution"  # Needs monitoring
    EMERGENCY = "emergency"  # Immediate human transfer


@dataclass
class SafetyResult:
    """Result of safety classification."""

    level: SafetyLevel
    triggered_keywords: list[str]
    confidence: float
    should_transfer: bool
    message: str


class SafetyClassifier:
    """
    Multi-language emergency and safety keyword detector.

    Detects emergency situations across Hindi, Tamil, Telugu, and Kannada
    for immediate transfer to human operators.
    """

    # Emergency keywords requiring immediate human transfer
    EMERGENCY_KEYWORDS = {
        # English
        "en": [
            "emergency",
            "accident",
            "heart attack",
            "stroke",
            "bleeding",
            "unconscious",
            "not breathing",
            "chest pain",
            "suicide",
            "dying",
            "ambulance",
            "icu",
            "critical",
        ],
        # Hindi
        "hi": [
            "इमरजेंसी",
            "दुर्घटना",
            "हार्ट अटैक",
            "दिल का दौरा",
            "खून",
            "बेहोश",
            "सांस नहीं",
            "छाती में दर्द",
            "सीने में दर्द",
            "मर रहा",
            "एंबुलेंस",
            "आईसीयू",
            "गंभीर",
            "एक्सीडेंट",
        ],
        # Tamil
        "ta": [
            "அவசரம்",
            "விபத்து",
            "மாரடைப்பு",
            "இரத்தப்போக்கு",
            "மயக்கம்",
            "மூச்சு விடவில்லை",
            "நெஞ்சு வலி",
            "ஆம்புலன்ஸ்",
            "தீவிர சிகிச்சை",
            "ஐசியு",
        ],
        # Telugu
        "te": [
            "అత్యవసర",
            "ప్రమాదం",
            "గుండెపోటు",
            "రక్తస్రావం",
            "స్పృహ లేదు",
            "ఊపిరి ఆడటం లేదు",
            "ఛాతీ నొప్పి",
            "అంబులెన్స్",
            "ఐసీయూ",
            "క్రిటికల్",
        ],
        # Kannada
        "kn": [
            "ತುರ್ತು",
            "ಅಪಘಾತ",
            "ಹೃದಯಾಘಾತ",
            "ರಕ್ತಸ್ರಾವ",
            "ಪ್ರಜ್ಞಾಹೀನ",
            "ಉಸಿರಾಡುತ್ತಿಲ್ಲ",
            "ಎದೆ ನೋವು",
            "ಆಂಬುಲೆನ್ಸ್",
            "ಐಸಿಯು",
            "ಗಂಭೀರ",
        ],
    }

    # Caution keywords requiring monitoring
    CAUTION_KEYWORDS = {
        "en": ["pain", "fever", "vomiting", "dizzy", "faint", "weak", "blood pressure"],
        "hi": ["दर्द", "बुखार", "उल्टी", "चक्कर", "कमज़ोर", "ब्लड प्रेशर"],
        "ta": ["வலி", "காய்ச்சல்", "வாந்தி", "தலைச்சுற்று", "பலவீனம்"],
        "te": ["నొప్పి", "జ్వరం", "వాంతి", "తలతిరుగుట", "బలహీనత"],
        "kn": ["ನೋವು", "ಜ್ವರ", "ವಾಂತಿ", "ತಲೆಸುತ್ತು", "ದೌರ್ಬಲ್ಯ"],
    }

    # Unsafe medical advice patterns (the assistant must not give these)
    UNSAFE_PATTERNS = [
        r"\b(?:prescribe[sd]?|prescription|dosage|dose)\b",
        r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg)\b",
        r"\b(?:take|taking)\b.{0,40}\b(?:tablets?|pills?|medicines?|capsules?)\b",
        r"\bdiagnos(?:is|e|ed)\b",
    ]

    def __init__(self) -> None:
        self._emergency_patterns = self._compile_keywords(self.EMERGENCY_KEYWORDS)
        self._caution_patterns = self._compile_keywords(self.CAUTION_KEYWORDS)
        self._unsafe_pattern = re.compile("|".join(self.UNSAFE_PATTERNS), re.IGNORECASE)

    @staticmethod
    def _keyword_pattern(keyword: str) -> re.Pattern[str]:
        """
        Build a matcher for one keyword.

        ASCII keywords need word boundaries so that e.g. "icu" does not match
        "ridiculous" or "pain" match "Spain". Indic-script keywords are matched
        as substrings because inflected/agglutinated forms are common.
        """
        escaped = re.escape(keyword)
        if keyword.isascii():
            return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        return re.compile(escaped)

    @classmethod
    def _compile_keywords(
        cls, keywords_by_lang: dict[str, list[str]]
    ) -> list[tuple[str, re.Pattern[str]]]:
        unique = sorted({k.lower() for words in keywords_by_lang.values() for k in words})
        return [(k, cls._keyword_pattern(k)) for k in unique]

    @staticmethod
    def _matches(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
        return [keyword for keyword, pattern in patterns if pattern.search(text)]

    def classify(self, text: str) -> SafetyResult:
        """
        Classify text for safety concerns.

        Emergency keywords take precedence over caution keywords.

        Args:
            text: Input text (patient query or model response).
        """
        emergency_matches = self._matches(text, self._emergency_patterns)
        if emergency_matches:
            return SafetyResult(
                level=SafetyLevel.EMERGENCY,
                triggered_keywords=emergency_matches,
                confidence=1.0,
                should_transfer=True,
                message="Emergency detected. Transferring to human operator immediately.",
            )

        caution_matches = self._matches(text, self._caution_patterns)
        if caution_matches:
            return SafetyResult(
                level=SafetyLevel.CAUTION,
                triggered_keywords=caution_matches,
                confidence=0.7,
                should_transfer=False,
                message="Medical concern detected. Monitoring conversation.",
            )

        return SafetyResult(
            level=SafetyLevel.SAFE,
            triggered_keywords=[],
            confidence=1.0,
            should_transfer=False,
            message="No safety concerns detected.",
        )

    def check_response_safety(self, response: str) -> tuple[bool, str]:
        """
        Check if model response contains unsafe medical advice.

        Args:
            response: Model's generated response

        Returns:
            (is_safe, reason) tuple
        """
        if self._unsafe_pattern.search(response):
            return False, "Response may contain medical advice. Needs review."
        return True, "Response is safe."

    def get_transfer_message(self, lang: str = "en") -> str:
        """Get human transfer message in specified language."""
        messages = {
            "en": "I'm connecting you with a healthcare professional right away. Please stay on the line.",
            "hi": "मैं आपको तुरंत एक स्वास्थ्य विशेषज्ञ से जोड़ रहा हूं। कृपया लाइन पर रहें।",
            "ta": "நான் உங்களை உடனடியாக ஒரு சுகாதார நிபுணருடன் இணைக்கிறேன். தயவுசெய்து காத்திருங்கள்.",
            "te": "నేను మిమ్మల్ని వెంటనే ఒక ఆరోగ్య నిపుణుడితో అనుసంధానం చేస్తున్నాను. దయచేసి వేచి ఉండండి.",
            "kn": "ನಾನು ನಿಮ್ಮನ್ನು ತಕ್ಷಣ ಆರೋಗ್ಯ ತಜ್ಞರೊಂದಿಗೆ ಸಂಪರ್ಕಿಸುತ್ತೇನೆ. ದಯವಿಟ್ಟು ಕಾಯಿರಿ.",
        }
        return messages.get(lang, messages["en"])


_default_classifier: SafetyClassifier | None = None


def check_safety(text: str) -> SafetyResult:
    """Quick safety check using a shared classifier instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SafetyClassifier()
    return _default_classifier.classify(text)
