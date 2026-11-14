"""
Cascaded speech-to-speech pipeline (STT -> LLM -> TTS).

This is the *baseline* alongside the unified SNAC + AudioLLM model:

- STT: Whisper (multilingual)
- LLM: Sarvam-1 2B
- TTS: Facebook MMS-TTS (VITS), one checkpoint per language

Usage::

    pipeline = SpeechPipeline()
    pipeline.load_models()
    audio_out, transcript, reply, metrics = pipeline.process(audio_in, language="hi")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from apollo_voice_engine.safety.classifier import SafetyClassifier

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampling of a mono float waveform."""
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.size / src_rate
    n_out = max(int(round(duration * dst_rate)), 1)
    x_old = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


@dataclass
class PipelineConfig:
    """Configuration for the cascaded speech pipeline."""

    whisper_model: str = "openai/whisper-small"
    llm_model: str = "sarvamai/sarvam-1"
    max_new_tokens: int = 100
    temperature: float = 0.7
    # MMS-TTS checkpoint per language (VITS models, no auth required)
    tts_models: dict[str, str] = field(
        default_factory=lambda: {
            "hi": "facebook/mms-tts-hin",
            "ta": "facebook/mms-tts-tam",
            "te": "facebook/mms-tts-tel",
            "kn": "facebook/mms-tts-kan",
            "en": "facebook/mms-tts-eng",
        }
    )
    device: str = field(default_factory=_default_device)


@dataclass
class PipelineMetrics:
    """Latency metrics for pipeline components."""

    stt_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    safety_triggered: bool = False


class SpeechPipeline:
    """End-to-end cascaded speech-to-speech pipeline for Indian languages."""

    LANGUAGE_CODES = {
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "kn": "Kannada",
        "en": "English",
    }

    TRANSFER_MESSAGES = {
        "hi": "आपातकाल का पता चला। मैं आपको तुरंत स्वास्थ्य विशेषज्ञ से जोड़ रहा हूं।",
        "ta": "அவசரநிலை கண்டறியப்பட்டது. சுகாதார நிபுணரை இணைக்கிறேன்.",
        "te": "అత్యవసర పరిస్థితి. ఆరోగ్య నిపుణుడిని కనెక్ట్ చేస్తున్నాను.",
        "kn": "ತುರ್ತು ಪರಿಸ್ಥಿತಿ. ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇನೆ.",
        "en": "Emergency detected. Connecting you with a healthcare professional.",
    }

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.safety = SafetyClassifier()

        self.whisper: Any = None
        self.whisper_processor: Any = None
        self.llm: Any = None
        self.llm_tokenizer: Any = None
        # language code -> (model, tokenizer); loaded lazily per language
        self._tts: dict[str, tuple[Any, Any]] = {}
        self.is_loaded = False

    @property
    def _dtype(self) -> torch.dtype:
        return torch.float16 if self.config.device == "cuda" else torch.float32

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_models(
        self,
        load_whisper: bool = True,
        load_llm: bool = True,
        tts_languages: tuple[str, ...] = ("hi",),
    ) -> None:
        """Load the pipeline models. TTS for other languages loads on demand."""
        logger.info("Loading speech pipeline models...")
        if load_whisper:
            self._load_whisper()
        if load_llm:
            self._load_llm()
        for lang in tts_languages:
            self._load_tts(lang)
        self.is_loaded = True
        logger.info("Speech pipeline models loaded")

    def _load_whisper(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper: %s", self.config.whisper_model)
        self.whisper_processor = WhisperProcessor.from_pretrained(self.config.whisper_model)
        self.whisper = (
            WhisperForConditionalGeneration.from_pretrained(
                self.config.whisper_model, dtype=self._dtype
            )
            .to(self.config.device)
            .eval()
        )

    def _load_llm(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM: %s", self.config.llm_model)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model, dtype=self._dtype, device_map=self.config.device
        ).eval()

    def _load_tts(self, language: str) -> tuple[Any, Any]:
        if language in self._tts:
            return self._tts[language]
        model_name = self.config.tts_models.get(language)
        if model_name is None:
            raise ValueError(f"No TTS model configured for language '{language}'")

        from transformers import AutoTokenizer, VitsModel

        logger.info("Loading TTS for %s: %s", language, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name).to(self.config.device).eval()
        self._tts[language] = (model, tokenizer)
        return self._tts[language]

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def transcribe(
        self, audio: np.ndarray, language: str = "hi", sample_rate: int = WHISPER_SAMPLE_RATE
    ) -> tuple[str, float]:
        """
        Transcribe audio with Whisper.

        Args:
            audio: Mono (or ``(channels, T)``) float waveform.
            language: Language code (hi, ta, te, kn, en).
            sample_rate: Sample rate of ``audio``; resampled to 16 kHz if needed.

        Returns:
            ``(text, latency_ms)``
        """
        if self.whisper is None:
            raise RuntimeError("Whisper not loaded. Call load_models() first.")

        start = time.perf_counter()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = resample(audio, sample_rate, WHISPER_SAMPLE_RATE)

        features = self.whisper_processor(
            audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.config.device, dtype=self.whisper.dtype)

        generated_ids = self.whisper.generate(
            features,
            language=self.LANGUAGE_CODES.get(language, "Hindi").lower(),
            task="transcribe",
            max_new_tokens=256,
        )
        text = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()

        latency = (time.perf_counter() - start) * 1000
        logger.debug("STT: %r (%.0fms)", text, latency)
        return text, latency

    @torch.inference_mode()
    def generate_response(self, text: str, language: str = "hi") -> tuple[str, float, bool]:
        """
        Generate a reply with the LLM, short-circuiting on emergencies.

        Returns:
            ``(response_text, latency_ms, is_emergency)``
        """
        start = time.perf_counter()

        if self.safety.classify(text).should_transfer:
            latency = (time.perf_counter() - start) * 1000
            return self.get_transfer_message(language), latency, True

        if self.llm is None:
            raise RuntimeError("LLM not loaded. Call load_models() first.")

        prompt = f"Patient: {text}\nApollo Assistant:"
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.config.device)
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        response = self.llm_tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = response.strip().split("\n")[0]

        latency = (time.perf_counter() - start) * 1000
        logger.debug("LLM: %r (%.0fms)", response, latency)
        return response, latency, False

    @torch.inference_mode()
    def synthesize(self, text: str, language: str = "hi") -> tuple[np.ndarray, int, float]:
        """
        Synthesize speech with MMS-TTS.

        Returns:
            ``(waveform, sample_rate, latency_ms)``
        """
        start = time.perf_counter()
        model, tokenizer = self._load_tts(language)
        inputs = tokenizer(text, return_tensors="pt").to(self.config.device)
        waveform = model(**inputs).waveform.squeeze().float().cpu().numpy()
        latency = (time.perf_counter() - start) * 1000
        logger.debug("TTS: %d samples (%.0fms)", waveform.size, latency)
        return waveform, model.config.sampling_rate, latency

    def process(
        self, audio_in: np.ndarray, language: str = "hi", sample_rate: int = WHISPER_SAMPLE_RATE
    ) -> tuple[np.ndarray, int, str, str, PipelineMetrics]:
        """
        Run STT -> LLM -> TTS.

        Returns:
            ``(audio_out, audio_out_sample_rate, transcription, response, metrics)``
        """
        metrics = PipelineMetrics()
        total_start = time.perf_counter()

        transcription, metrics.stt_ms = self.transcribe(audio_in, language, sample_rate)
        response, metrics.llm_ms, metrics.safety_triggered = self.generate_response(
            transcription, language
        )
        audio_out, out_rate, metrics.tts_ms = self.synthesize(response, language)

        metrics.total_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "Pipeline: STT=%.0fms, LLM=%.0fms, TTS=%.0fms, Total=%.0fms",
            metrics.stt_ms,
            metrics.llm_ms,
            metrics.tts_ms,
            metrics.total_ms,
        )
        return audio_out, out_rate, transcription, response, metrics

    def get_transfer_message(self, language: str) -> str:
        """Emergency transfer message in ``language`` (English fallback)."""
        return self.TRANSFER_MESSAGES.get(language, self.TRANSFER_MESSAGES["en"])
