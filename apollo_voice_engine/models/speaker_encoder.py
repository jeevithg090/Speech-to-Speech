"""
Speaker Encoder for Voice Identity Conditioning.

This module provides the mechanism to encode voice identity into a fixed-size embedding.
This embedding is used to condition the AudioLLM generation, ensuring voice consistency
and enabling "Session Anchoring".
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SpeakerEncoder(nn.Module):
    """
    Encodes audio into a speaker embedding vector.

    In a production system, this would wrap a model like Resemblyzer, WavLM, or ECAPA-TDNN.
    For this implementation, we provide the interface and a learnable/random fallback
    to validate the pipeline architecture.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        device: str = "cpu",
        model_name: str = "mock_encoder",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.model_name = model_name

        # In a real scenario, load the model here
        # self.model = load_pretrained(model_name)

        logger.info(f"Initialized SpeakerEncoder (dim={embedding_dim}, device={device})")

    def forward(self, audio: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Encode audio samples into a unit-norm speaker embedding.

        Mock implementation: the embedding is a pseudo-random vector seeded by
        the audio content, so identical audio always maps to the same "voice"
        while different audio maps to (almost surely) different ones.

        Args:
            audio: Audio samples shaped ``(T,)`` or ``(B, T)``.

        Returns:
            Tensor of shape ``(1, embedding_dim)``.
        """
        audio_t = torch.as_tensor(np.asarray(audio) if isinstance(audio, np.ndarray) else audio)
        audio_t = audio_t.detach().to("cpu", dtype=torch.float64)
        # In a real encoder: embedding = self.model(audio)
        seed = int(audio_t.abs().sum().item() * 1000) % 2**32
        generator = torch.Generator().manual_seed(seed)
        embedding = torch.randn(1, self.embedding_dim, generator=generator)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.to(self.device)

    def get_random_embedding(self) -> torch.Tensor:
        """Generate a random speaker embedding for testing."""
        embedding = torch.randn(1, self.embedding_dim, device=self.device)
        return nn.functional.normalize(embedding, p=2, dim=1)
