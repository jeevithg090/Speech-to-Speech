"""
Inference Engine with Voice Activity Detection

Real-time inference pipeline with:
- Voice Activity Detection (VAD) for turn-taking
- Streaming audio processing
- Low-latency response generation
"""

import logging
import queue
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from apollo_voice_engine.models.snac_wrapper import SNACWrapper

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    aggressiveness: int = 2  # 0-3, higher = more aggressive
    frame_duration_ms: int = 30
    padding_duration_ms: int = 300
    sample_rate: int = 24000


@dataclass
class InferenceConfig:
    """Inference engine configuration."""

    max_ttft_ms: int = 250  # Time to first token target
    max_e2e_latency_ms: int = 500  # End-to-end target
    max_new_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9


class VoiceActivityDetector:
    """
    Voice Activity Detection for turn-taking in conversations.

    Wraps WebRTC VAD, which only accepts 16-bit PCM at 8/16/32/48 kHz in
    10/20/30 ms frames. Audio at other rates (e.g. SNAC's 24 kHz) is
    resampled to 16 kHz, and buffers of arbitrary length are split into
    frames; a buffer counts as speech if any full frame contains speech.
    """

    SUPPORTED_RATES = (8000, 16000, 32000, 48000)
    SUPPORTED_FRAME_MS = (10, 20, 30)

    def __init__(self, config: VADConfig):
        if config.frame_duration_ms not in self.SUPPORTED_FRAME_MS:
            raise ValueError(f"frame_duration_ms must be one of {self.SUPPORTED_FRAME_MS}")
        if not 0 <= config.aggressiveness <= 3:
            raise ValueError("aggressiveness must be between 0 and 3")
        self.config = config
        self.vad = None
        self._init_vad()

    def _init_vad(self) -> None:
        """Initialize WebRTC VAD (optional dependency)."""
        try:
            import webrtcvad
        except ImportError:
            logger.warning("webrtcvad not installed. VAD disabled (all audio treated as speech).")
            self.vad = None
            return
        self.vad = webrtcvad.Vad(self.config.aggressiveness)
        logger.info("VAD initialized with aggressiveness=%d", self.config.aggressiveness)

    @staticmethod
    def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Linear-interpolation resampling (adequate for VAD decisions)."""
        if src_rate == dst_rate or samples.size == 0:
            return samples
        duration = samples.size / src_rate
        n_out = max(int(round(duration * dst_rate)), 1)
        x_old = np.linspace(0.0, duration, num=samples.size, endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.int16)

    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Check whether a buffer of 16-bit mono PCM contains speech.

        Args:
            audio_frame: Raw little-endian int16 PCM at ``config.sample_rate``.
        """
        if self.vad is None:
            return True

        samples = np.frombuffer(audio_frame, dtype=np.int16)
        rate = self.config.sample_rate
        if rate not in self.SUPPORTED_RATES:
            samples = self._resample(samples, rate, 16000)
            rate = 16000

        frame_len = rate * self.config.frame_duration_ms // 1000
        if samples.size < frame_len:
            return False

        for start in range(0, samples.size - frame_len + 1, frame_len):
            frame = samples[start : start + frame_len].tobytes()
            if self.vad.is_speech(frame, rate):
                return True
        return False

    def detect_end_of_speech(
        self,
        audio_stream: Iterable[bytes],
        on_speech_end: Callable[[], None],
    ) -> Generator[bytes, None, None]:
        """
        Pass frames through while calling ``on_speech_end`` after each utterance.

        The callback fires once per utterance, after ``padding_duration_ms`` of
        silence following at least one speech frame.
        """
        silence_frames = 0
        heard_speech = False
        padding_frames = max(self.config.padding_duration_ms // self.config.frame_duration_ms, 1)

        for frame in audio_stream:
            yield frame

            if self.is_speech(frame):
                silence_frames = 0
                heard_speech = True
                continue

            silence_frames += 1
            if heard_speech and silence_frames >= padding_frames:
                logger.debug("End of speech detected")
                on_speech_end()
                silence_frames = 0
                heard_speech = False


class InferenceEngine:
    """
    Real-time inference engine for the Apollo Voice Engine.

    Features:
    - Streaming audio input processing
    - VAD-triggered generation
    - Latency monitoring
    - Concurrent request handling
    """

    MIN_AUDIO_TOKENS = SNACWrapper.TOKENS_PER_FRAME

    def __init__(
        self,
        model,  # AudioLLM instance
        snac_wrapper,  # SNACWrapper instance
        inference_config: InferenceConfig | None = None,
        vad_config: VADConfig | None = None,
    ):
        self.model = model
        self.snac = snac_wrapper
        self.inference_config = inference_config or InferenceConfig()
        self.vad = VoiceActivityDetector(vad_config or VADConfig())

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "avg_ttft_ms": 0.0,
            "avg_e2e_latency_ms": 0.0,
        }

        # Audio buffer for streaming
        self.audio_buffer: queue.Queue[np.ndarray] = queue.Queue()

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Add audio chunk to processing buffer.

        Args:
            audio_chunk: Audio samples (float32, 24kHz)
        """
        self.audio_buffer.put(audio_chunk)

    def _collect_audio(self, timeout: float = 5.0) -> torch.Tensor | None:
        """Collect buffered audio into a single tensor."""
        chunks = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                chunk = self.audio_buffer.get(timeout=0.1)
                chunks.append(chunk)

                # Convert to bytes for VAD
                audio_bytes = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                if not self.vad.is_speech(audio_bytes):
                    # End of speech detected
                    break

            except queue.Empty:
                if chunks:
                    break

        if not chunks:
            return None

        return torch.tensor(np.concatenate(chunks), dtype=torch.float32)

    def generate_response(
        self, audio_input: torch.Tensor, text_context: str | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Generate a response from audio input.

        Args:
            audio_input: Input audio tensor
            text_context: Optional text context/prompt

        Returns:
            Tuple of (response_audio, metrics)
        """
        start_time = time.perf_counter()
        self.metrics["total_requests"] += 1

        # Encode input audio to tokens
        encode_start = time.perf_counter()
        audio_tokens = self.snac.encode(audio_input)
        encode_time = (time.perf_counter() - encode_start) * 1000

        # Prepare model input
        input_ids = self.model.build_audio_input_ids(audio_tokens, text_prompt=text_context)
        prompt_len = input_ids.shape[1]

        # Generate response
        generate_start = time.perf_counter()
        output_ids = self.model.generate_response(
            input_ids=input_ids,
            max_new_tokens=self.inference_config.max_new_tokens,
            temperature=self.inference_config.temperature,
            top_p=self.inference_config.top_p,
        )
        generate_time = (time.perf_counter() - generate_start) * 1000

        # generate() with input_ids returns prompt + continuation
        new_ids = output_ids[:, prompt_len:]
        num_new = max(new_ids.shape[1], 1)
        # Approximate time to first token as encode time + mean per-token decode time
        ttft = encode_time + generate_time / num_new

        response_tokens = self.model.extract_audio_tokens(new_ids)
        if response_tokens is not None and response_tokens.shape[-1] < self.MIN_AUDIO_TOKENS:
            logger.warning(
                "Generated %d audio tokens; at least one SNAC frame (%d) is needed",
                response_tokens.shape[-1],
                self.MIN_AUDIO_TOKENS,
            )
            response_tokens = None

        if response_tokens is not None:
            decode_start = time.perf_counter()
            response_audio = self.snac.decode(response_tokens)
            decode_time = (time.perf_counter() - decode_start) * 1000
        else:
            response_audio = None
            decode_time = 0.0

        e2e_latency = (time.perf_counter() - start_time) * 1000

        n = self.metrics["total_requests"]
        self.metrics["avg_ttft_ms"] = (self.metrics["avg_ttft_ms"] * (n - 1) + ttft) / n
        self.metrics["avg_e2e_latency_ms"] = (
            self.metrics["avg_e2e_latency_ms"] * (n - 1) + e2e_latency
        ) / n

        request_metrics = {
            "encode_time_ms": encode_time,
            "generate_time_ms": generate_time,
            "decode_time_ms": decode_time,
            "ttft_ms": ttft,
            "e2e_latency_ms": e2e_latency,
            "input_tokens": prompt_len,
            "output_tokens": new_ids.shape[1],
            "meets_ttft_target": ttft < self.inference_config.max_ttft_ms,
            "meets_latency_target": e2e_latency < self.inference_config.max_e2e_latency_ms,
        }

        logger.info(
            f"Request completed: TTFT={ttft:.1f}ms, E2E={e2e_latency:.1f}ms, "
            f"Tokens={request_metrics['output_tokens']}"
        )

        return response_audio, request_metrics

    def stream_response(
        self, audio_input: torch.Tensor, chunk_size: int = 8192
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream response audio in chunks for real-time playback.

        Args:
            audio_input: Input audio tensor
            chunk_size: Samples per chunk

        Yields:
            Audio chunks as numpy arrays
        """
        response_audio, _ = self.generate_response(audio_input)

        if response_audio is None:
            return

        audio_np = response_audio.cpu().numpy()

        for i in range(0, len(audio_np), chunk_size):
            yield audio_np[i : i + chunk_size]

    def get_metrics(self) -> dict:
        """Get current inference metrics."""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset inference metrics."""
        self.metrics = {
            "total_requests": 0,
            "avg_ttft_ms": 0.0,
            "avg_e2e_latency_ms": 0.0,
        }


class StreamingSession:
    """
    Manages a streaming conversation session.

    Handles continuous audio input with VAD-based turn-taking.
    """

    def __init__(self, engine: InferenceEngine, session_id: str, language: str = "hi"):
        self.engine = engine
        self.session_id = session_id
        self.language = language

        self.is_active = False
        self.turn_count = 0
        self._audio_buffer = []
        self._response_callback = None

    def start(self, on_response: Callable[[np.ndarray], None]):
        """Start the streaming session."""
        self.is_active = True
        self._response_callback = on_response
        logger.info(f"Session {self.session_id} started")

    def stop(self):
        """Stop the streaming session."""
        self.is_active = False
        logger.info(f"Session {self.session_id} stopped after {self.turn_count} turns")

    def on_audio_chunk(self, chunk: np.ndarray):
        """Process an incoming audio chunk."""
        if not self.is_active:
            return

        self._audio_buffer.append(chunk)

    def on_speech_end(self):
        """Called when VAD detects end of speech."""
        if not self._audio_buffer:
            return

        # Combine buffered audio
        audio = torch.tensor(np.concatenate(self._audio_buffer), dtype=torch.float32)
        self._audio_buffer = []
        self.turn_count += 1

        # Generate and deliver response
        for chunk in self.engine.stream_response(audio):
            if self._response_callback and self.is_active:
                self._response_callback(chunk)
