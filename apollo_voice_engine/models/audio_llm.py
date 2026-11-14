"""
Extended Audio-LLM model.

Extends Sarvam-1 2B with SNAC audio tokens for unified speech-to-speech
generation: the model processes audio tokens alongside text tokens and
predicts both with a single next-token objective.

Vocabulary layout after :meth:`AudioLLM.load_model`::

    [0, audio_offset)                         text tokens + <|audio_start|>/<|audio_end|>
    [audio_offset, audio_offset + 4096)       SNAC codebook entries 0..4095
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AudioLLM(nn.Module):
    """
    Sarvam-1 causal LM with an audio-token vocabulary extension.

    Architecture:
    - Base: Sarvam-1 2B (28 layers, 2048 hidden, 16 heads)
    - Extended vocabulary: +4096 SNAC audio tokens
    - Optional speaker embedding projected into a prefix "token"
    """

    # Offset used by :class:`SNACWrapper` for its *standalone* token ids. It is
    # stripped before tokens are mapped into this model's vocabulary.
    AUDIO_TOKEN_OFFSET = 50000
    AUDIO_VOCAB_SIZE = 4096

    AUDIO_START_TOKEN = "<|audio_start|>"
    AUDIO_END_TOKEN = "<|audio_end|>"

    SPEAKER_DIM = 512

    def __init__(
        self,
        model_name: str = "sarvamai/sarvam-1",
        device: str | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Args:
            model_name: Hugging Face model id or local checkpoint directory.
            device: Device to load the model on (auto-detected when ``None``).
            load_in_8bit: Load with bitsandbytes 8-bit quantization (CUDA only).
            load_in_4bit: Load with bitsandbytes 4-bit quantization (CUDA only).
        """
        super().__init__()
        if load_in_8bit and load_in_4bit:
            raise ValueError("Choose at most one of load_in_8bit / load_in_4bit")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        # Populated by load_model()
        self.model: Any = None
        self.tokenizer: Any = None
        self.original_vocab_size: int | None = None
        self.audio_offset: int | None = None
        self.extended_vocab_size: int | None = None

        self.speaker_dim = self.SPEAKER_DIM
        self.speaker_proj: nn.Linear | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load the base model and extend it with the audio vocabulary."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading base model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.AUDIO_START_TOKEN, self.AUDIO_END_TOKEN]}
        )

        load_kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
        if self.load_in_8bit or self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit, load_in_4bit=self.load_in_4bit
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self._extend_vocabulary()

        self.speaker_proj = nn.Linear(self.speaker_dim, self.model.config.hidden_size).to(
            self.device, dtype=self.model.dtype
        )
        logger.info(
            "Model loaded. Vocab extended: %s -> %s",
            self.original_vocab_size,
            self.extended_vocab_size,
        )

    def configure_vocabulary(self, text_vocab_size: int) -> None:
        """
        Set the audio-token offset for a given text vocabulary size.

        Exposed separately so the id mapping can be used (and tested) without
        loading model weights.
        """
        self.audio_offset = text_vocab_size
        self.extended_vocab_size = text_vocab_size + self.AUDIO_VOCAB_SIZE

    def _extend_vocabulary(self) -> None:
        """Resize embeddings so audio tokens follow the text + special tokens."""
        self.original_vocab_size = self.model.config.vocab_size
        tokenizer_size = len(self.tokenizer)  # includes the audio special tokens

        if self.original_vocab_size >= tokenizer_size + self.AUDIO_VOCAB_SIZE:
            # Checkpoint written by save_pretrained() already has the audio rows.
            self.configure_vocabulary(self.original_vocab_size - self.AUDIO_VOCAB_SIZE)
            logger.info("Checkpoint already contains audio vocabulary")
            return

        # Special tokens may be assigned ids past the model's original vocab size,
        # and some models pad their embedding matrix beyond the tokenizer size.
        text_vocab = max(tokenizer_size, self.original_vocab_size)
        self.configure_vocabulary(text_vocab)
        self.model.resize_token_embeddings(self.extended_vocab_size)

        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()
            nn.init.normal_(embeddings.weight[self.audio_offset :], mean=0.0, std=0.02)
        logger.info("Extended embeddings for %d audio tokens", self.AUDIO_VOCAB_SIZE)

    def _require_vocab(self) -> int:
        if self.audio_offset is None:
            raise RuntimeError("Vocabulary not configured; call load_model() first")
        return self.audio_offset

    # ------------------------------------------------------------------
    # Token id mapping
    # ------------------------------------------------------------------
    @property
    def audio_start_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.AUDIO_START_TOKEN)

    @property
    def audio_end_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.AUDIO_END_TOKEN)

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Tokenize text to a ``(1, N)`` id tensor."""
        return self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(
            self.device
        )

    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """Decode the text tokens in ``tokens``; audio tokens are skipped."""
        offset = self._require_vocab()
        text_tokens = tokens[tokens < offset]
        return self.tokenizer.decode(text_tokens, skip_special_tokens=True)

    def audio_tokens_to_ids(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Map SNAC codes to extended-vocabulary ids.

        Accepts raw codes (0-4095) or :class:`SNACWrapper` output, which carries
        ``AUDIO_TOKEN_OFFSET``.
        """
        offset = self._require_vocab()
        audio_tokens = audio_tokens.long()
        if audio_tokens.numel() and audio_tokens.min() >= self.AUDIO_TOKEN_OFFSET:
            audio_tokens = audio_tokens - self.AUDIO_TOKEN_OFFSET
        if audio_tokens.numel() and (
            audio_tokens.min() < 0 or audio_tokens.max() >= self.AUDIO_VOCAB_SIZE
        ):
            raise ValueError("Audio tokens must be in [0, 4096) (optionally offset by 50000)")
        return audio_tokens + offset

    def ids_to_audio_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map extended-vocabulary ids back to SNACWrapper-style tokens (code + 50000)."""
        offset = self._require_vocab()
        return token_ids - offset + self.AUDIO_TOKEN_OFFSET

    def is_audio_token(self, token_id: int) -> bool:
        """Whether ``token_id`` is one of the 4096 audio tokens."""
        offset = self._require_vocab()
        return offset <= token_id < offset + self.AUDIO_VOCAB_SIZE

    def build_audio_input_ids(
        self, audio_tokens: torch.Tensor, text_prompt: str | None = None
    ) -> torch.Tensor:
        """
        Build ``[text_prompt] <|audio_start|> audio... <|audio_end|>`` as a ``(1, N)`` tensor.
        """
        parts: list[torch.Tensor] = []
        if text_prompt:
            parts.append(self.text_to_tokens(text_prompt).squeeze(0))

        audio_ids = self.audio_tokens_to_ids(audio_tokens.to(self.device)).reshape(-1)
        parts.append(torch.tensor([self.audio_start_id], device=self.device))
        parts.append(audio_ids)
        parts.append(torch.tensor([self.audio_end_id], device=self.device))
        return torch.cat(parts, dim=0).unsqueeze(0)

    def prepare_audio_input(
        self,
        audio_tokens: torch.Tensor,
        text_prompt: str | None = None,
        speaker_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build input embeddings with optional speaker conditioning.

        The projected speaker embedding is prepended as a prefix position
        ("session anchor") ahead of the prompt and audio tokens.

        Returns:
            ``(inputs_embeds, attention_mask)``
        """
        input_ids = self.build_audio_input_ids(audio_tokens, text_prompt)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if speaker_embedding is not None and self.speaker_proj is not None:
            embed_input = speaker_embedding.to(device=self.device, dtype=inputs_embeds.dtype)
            speaker_cond = self.speaker_proj(embed_input)
            if speaker_cond.dim() == 2:
                speaker_cond = speaker_cond.unsqueeze(1)
            logger.debug("Injecting speaker embedding, shape=%s", tuple(speaker_cond.shape))
            inputs_embeds = torch.cat([speaker_cond, inputs_embeds], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        return inputs_embeds, attention_mask

    # ------------------------------------------------------------------
    # Forward / generation
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to the underlying causal LM."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    @torch.no_grad()
    def generate_response(
        self,
        input_ids: torch.Tensor | None = None,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        override_output_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate a continuation from ``input_ids`` or ``inputs_embeds``.

        When ``inputs_embeds`` is given, Hugging Face returns only the newly
        generated ids; with ``input_ids`` the prompt is included.

        Args:
            override_output_ids: Return these ids without running the model
                (useful for demos with pre-computed responses).
        """
        if override_output_ids is not None:
            return override_output_ids.to(self.device)

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        if attention_mask is None:
            shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
            attention_mask = torch.ones(shape, dtype=torch.long, device=self.device)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        gen_kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            **kwargs,
        }
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        if inputs_embeds is not None:
            gen_kwargs["inputs_embeds"] = inputs_embeds
        else:
            gen_kwargs["input_ids"] = input_ids

        return self.model.generate(**gen_kwargs)

    def extract_audio_tokens(self, generated_ids: torch.Tensor) -> torch.Tensor | None:
        """
        Extract the audio segment from a generated sequence.

        Looks for the *last* ``<|audio_start|>`` (so a prompt containing input
        audio is skipped) and reads until ``<|audio_end|>`` or the end of the
        sequence. Non-audio ids inside the segment are dropped.

        Returns:
            ``(1, N)`` SNACWrapper-style tokens, or ``None`` if no audio was found.
        """
        offset = self._require_vocab()
        tokens = generated_ids.reshape(-1).tolist()
        start_id, end_id = self.audio_start_id, self.audio_end_id

        # Index just past the last <|audio_start|> (0 if absent)
        start = len(tokens) - tokens[::-1].index(start_id) if start_id in tokens else 0
        segment = tokens[start:]
        if end_id in segment:
            segment = segment[: segment.index(end_id)]

        audio_ids = [t for t in segment if offset <= t < offset + self.AUDIO_VOCAB_SIZE]
        if not audio_ids:
            logger.warning("No audio tokens found in generated sequence")
            return None
        return self.ids_to_audio_tokens(torch.tensor([audio_ids], device=self.device))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pretrained(self, save_path: str) -> None:
        """Save the extended model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Model saved to %s", save_path)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: Any) -> AudioLLM:
        """Load a checkpoint previously written by :meth:`save_pretrained`."""
        instance = cls(model_name=model_path, **kwargs)
        instance.load_model()
        return instance
