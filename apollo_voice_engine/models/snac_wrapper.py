"""
SNAC (Multi-Scale Neural Audio Codec) wrapper.

Converts audio waveforms to discrete tokens (and back) so they can be placed
in the extended Sarvam-1 vocabulary.

The 24 kHz SNAC model produces three codebooks per "frame" with a 1:2:4 ratio
(coarse, medium, fine). A frame spans ``hop_length * 4 = 2048`` samples, i.e.
~11.7 frames per second. Codes are interleaved frame by frame into a single
flat sequence of ``TOKENS_PER_FRAME = 7`` tokens::

    [C0, M0, M1, F0, F1, F2, F3, C1, M2, M3, F4, ...]
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class SNACWrapper:
    """Thin, device-aware wrapper around the SNAC 24 kHz codec."""

    # Token offset in the extended vocabulary (kept for backwards compatibility
    # with checkpoints and notebooks that rely on it).
    AUDIO_TOKEN_OFFSET = 50000
    AUDIO_VOCAB_SIZE = 4096

    # Codebook ratio of the 24 kHz SNAC model: 1 coarse, 2 medium, 4 fine.
    SCALE_RATIOS: tuple[int, int, int] = (1, 2, 4)
    TOKENS_PER_FRAME = sum(SCALE_RATIOS)
    SAMPLES_PER_FRAME = 2048  # hop_length (512) * coarsest vq stride (4)

    def __init__(
        self,
        model_name: str = "hubertsiuzdak/snac_24khz",
        sample_rate: int = 24000,
        device: str | None = None,
    ):
        """
        Args:
            model_name: Hugging Face model id (or local directory) for SNAC.
            sample_rate: Audio sample rate expected by the codec.
            device: Device to run on; auto-detected when ``None``.
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any = None

    @property
    def frames_per_second(self) -> float:
        """Number of coarse frames per second of audio."""
        return self.sample_rate / self.SAMPLES_PER_FRAME

    def load_model(self) -> None:
        """Load the SNAC model from Hugging Face (or a local directory)."""
        try:
            from snac import SNAC
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("SNAC package not installed. Run: pip install snac") from exc

        logger.info("Loading SNAC model: %s", self.model_name)
        model = SNAC.from_pretrained(self.model_name)
        self.model = model.to(self.device).eval()
        logger.info("SNAC model loaded on %s", self.device)

    def ensure_loaded(self) -> None:
        """Load the model lazily on first use."""
        if self.model is None:
            self.load_model()

    @staticmethod
    def _to_codec_shape(audio: torch.Tensor) -> torch.Tensor:
        """Reshape ``(T,)``, ``(B, T)`` or ``(B, 1, T)`` audio to ``(B, 1, T)``."""
        if audio.dim() == 1:
            return audio.unsqueeze(0).unsqueeze(0)
        if audio.dim() == 2:
            return audio.unsqueeze(1)
        if audio.dim() == 3 and audio.shape[1] == 1:
            return audio
        raise ValueError(
            f"Expected mono audio shaped (T,), (B, T) or (B, 1, T); got {tuple(audio.shape)}"
        )

    @torch.no_grad()
    def encode(
        self, audio: torch.Tensor, flatten: bool = True
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Encode an audio waveform to discrete tokens.

        Args:
            audio: Mono audio shaped ``(T,)``, ``(B, T)`` or ``(B, 1, T)``.
            flatten: Interleave the multi-scale codes into one sequence.

        Returns:
            Flat ``(B, frames * 7)`` tensor, or the list of per-scale code
            tensors when ``flatten`` is ``False``. ``AUDIO_TOKEN_OFFSET`` is
            applied in both cases.
        """
        self.ensure_loaded()
        audio = self._to_codec_shape(audio.float()).to(self.device)
        codes = self.model.encode(audio)

        if flatten:
            return self._flatten_codes(codes) + self.AUDIO_TOKEN_OFFSET
        return [c + self.AUDIO_TOKEN_OFFSET for c in codes]

    @torch.no_grad()
    def decode(
        self, tokens: torch.Tensor | list[torch.Tensor], is_flattened: bool = True
    ) -> torch.Tensor:
        """
        Decode tokens back to an audio waveform.

        Args:
            tokens: Flat token tensor (``(N,)`` or ``(B, N)``) or a list of
                per-scale code tensors. The offset may or may not be applied.
            is_flattened: Whether ``tokens`` is a flat interleaved sequence.

        Returns:
            Audio tensor shaped ``(T,)`` for a single item, otherwise ``(B, T)``.
        """
        self.ensure_loaded()

        if is_flattened:
            if not isinstance(tokens, torch.Tensor):
                raise TypeError("Flattened tokens must be a tensor")
            flat = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
            codes = self._unflatten_codes(self._strip_offset(flat))
        else:
            codes = [self._strip_offset(c) for c in tokens]

        codes = [c.long().to(self.device) for c in codes]
        audio = self.model.decode(codes)  # (B, 1, T)
        audio = audio.squeeze(1)
        return audio.squeeze(0) if audio.shape[0] == 1 else audio

    def _strip_offset(self, tokens: torch.Tensor) -> torch.Tensor:
        """Remove ``AUDIO_TOKEN_OFFSET`` if present and clamp to the codebook range."""
        if tokens.numel() and tokens.min() >= self.AUDIO_TOKEN_OFFSET:
            tokens = tokens - self.AUDIO_TOKEN_OFFSET
        return tokens.clamp(0, self.AUDIO_VOCAB_SIZE - 1)

    def _flatten_codes(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """
        Interleave multi-scale codes frame by frame.

        Each scale is reshaped to ``(B, frames, ratio)`` and the three are
        concatenated on the last dimension, which is equivalent to the
        ``[C, M, M, F, F, F, F]`` pattern described in the module docstring.
        """
        if len(codes) != len(self.SCALE_RATIOS):
            raise ValueError(f"Expected {len(self.SCALE_RATIOS)} code scales, got {len(codes)}")

        batch, num_frames = codes[0].shape
        parts = []
        for scale, ratio in zip(codes, self.SCALE_RATIOS, strict=True):
            if scale.shape[1] != num_frames * ratio:
                raise ValueError(
                    f"Inconsistent SNAC code lengths: {[tuple(c.shape) for c in codes]}"
                )
            parts.append(scale.reshape(batch, num_frames, ratio))
        return torch.cat(parts, dim=-1).reshape(batch, num_frames * self.TOKENS_PER_FRAME)

    def _unflatten_codes(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Reverse :meth:`_flatten_codes`. Trailing partial frames are dropped."""
        batch = tokens.shape[0]
        num_frames = tokens.shape[1] // self.TOKENS_PER_FRAME
        if num_frames == 0:
            raise ValueError(
                f"Need at least {self.TOKENS_PER_FRAME} tokens to decode, got {tokens.shape[1]}"
            )
        frames = tokens[:, : num_frames * self.TOKENS_PER_FRAME].reshape(
            batch, num_frames, self.TOKENS_PER_FRAME
        )
        codes = []
        start = 0
        for ratio in self.SCALE_RATIOS:
            codes.append(frames[:, :, start : start + ratio].reshape(batch, num_frames * ratio))
            start += ratio
        return codes

    def get_audio_duration(self, num_tokens: int) -> float:
        """Duration in seconds represented by ``num_tokens`` flat tokens."""
        num_frames = num_tokens // self.TOKENS_PER_FRAME
        return num_frames / self.frames_per_second

    def get_token_count(self, duration_seconds: float) -> int:
        """Number of flat tokens SNAC produces for ``duration_seconds`` of audio."""
        num_samples = round(duration_seconds * self.sample_rate)
        num_frames = -(-num_samples // self.SAMPLES_PER_FRAME)  # ceil: SNAC right-pads
        return num_frames * self.TOKENS_PER_FRAME
