"""
Training-example construction for the AudioLLM.

Three phases, all trained with a plain causal-LM loss where prompt positions
are masked with ``IGNORE_INDEX``:

- ``understanding`` (ASR):   prompt + <as> audio <ae>        -> text
- ``generation``    (TTS):   prompt + text + <as>            -> audio <ae>
- ``end2end``       (S2S):   <as> audio <ae> <as>            -> audio <ae>
  (self-reconstruction until paired dialogue audio is available)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

IGNORE_INDEX = -100
PHASES = ("understanding", "generation", "end2end")


@dataclass
class TrainingExample:
    input_ids: torch.Tensor  # (N,)
    labels: torch.Tensor  # (N,)


def _text_ids(audio_llm: Any, text: str, max_length: int, add_special_tokens: bool) -> torch.Tensor:
    ids = audio_llm.tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"]
    return ids.reshape(-1).long()


def build_example(
    audio_llm: Any,
    audio_tokens: torch.Tensor,
    text: str,
    language: str,
    phase: str,
    max_text_tokens: int = 256,
) -> TrainingExample:
    """
    Build one ``(input_ids, labels)`` pair for ``phase``.

    Args:
        audio_llm: Loaded :class:`AudioLLM` (only its tokenizer and id mapping are used).
        audio_tokens: SNAC tokens for the utterance (raw or SNACWrapper-offset).
        text: Transcript of the utterance.
        language: Language code, used in the instruction prompt.
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")

    audio_ids = audio_llm.audio_tokens_to_ids(audio_tokens.reshape(-1).cpu())
    start = torch.tensor([audio_llm.audio_start_id], dtype=torch.long)
    end = torch.tensor([audio_llm.audio_end_id], dtype=torch.long)
    audio_segment = torch.cat([start, audio_ids, end])

    if phase == "understanding":
        prompt = _text_ids(
            audio_llm, f"Transcribe the following audio in {language}:", max_text_tokens, True
        )
        prefix = torch.cat([prompt, audio_segment])
        target = _text_ids(audio_llm, text, max_text_tokens, False)
        eos = audio_llm.tokenizer.eos_token_id
        if eos is not None:
            target = torch.cat([target, torch.tensor([eos], dtype=torch.long)])
    elif phase == "generation":
        prefix = torch.cat(
            [
                _text_ids(
                    audio_llm,
                    f"Synthesize the following text in {language}: {text}",
                    max_text_tokens,
                    True,
                ),
                start,
            ]
        )
        target = torch.cat([audio_ids, end])
    else:  # end2end
        prefix = torch.cat([audio_segment, start])
        target = torch.cat([audio_ids, end])

    input_ids = torch.cat([prefix, target])
    labels = torch.cat([torch.full((prefix.numel(),), IGNORE_INDEX, dtype=torch.long), target])
    return TrainingExample(input_ids=input_ids, labels=labels)


def pad_batch(examples: list[TrainingExample], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad examples into ``input_ids``, ``labels`` and ``attention_mask`` tensors."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    max_len = max(ex.input_ids.numel() for ex in examples)
    batch = len(examples)
    input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for i, ex in enumerate(examples):
        n = ex.input_ids.numel()
        input_ids[i, :n] = ex.input_ids
        labels[i, :n] = ex.labels
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate(batch: list[dict[str, Any]], audio_llm: Any, phase: str) -> dict[str, torch.Tensor]:
    """DataLoader ``collate_fn``: dataset items -> padded model inputs."""
    examples = [
        build_example(audio_llm, item["audio_tokens"], item["text"], item["language"], phase)
        for item in batch
    ]
    tokenizer = audio_llm.tokenizer
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    return pad_batch(examples, pad_id if pad_id is not None else 0)
