"""Tests for AudioLLM training-example construction."""

from __future__ import annotations

import pytest
import torch

from apollo_voice_engine.training import IGNORE_INDEX, PHASES, build_example, collate, pad_batch

AUDIO = torch.tensor([50000, 50001, 50002, 50003, 50004, 50005, 50006])


def supervised(example):
    mask = example.labels != IGNORE_INDEX
    return example.input_ids[mask]


def test_unknown_phase(audio_llm):
    with pytest.raises(ValueError):
        build_example(audio_llm, AUDIO, "hello", "hi", "bogus")


def test_understanding_supervises_text_only(audio_llm):
    ex = build_example(audio_llm, AUDIO, "hello world", "hi", "understanding")
    assert ex.input_ids.shape == ex.labels.shape
    hello, world = audio_llm.tokenizer.convert_tokens_to_ids(["hello", "world"])
    assert supervised(ex).tolist() == [hello, world, audio_llm.tokenizer.eos_token_id]
    # Audio segment is in the (masked) prompt, mapped to the extended vocabulary
    audio_ids = audio_llm.audio_tokens_to_ids(AUDIO).tolist()
    seq = ex.input_ids.tolist()
    start = seq.index(audio_llm.audio_start_id)
    assert seq[start + 1 : start + 1 + len(audio_ids)] == audio_ids


def test_generation_supervises_audio_and_end_marker(audio_llm):
    ex = build_example(audio_llm, AUDIO, "hello", "hi", "generation")
    expected = [*audio_llm.audio_tokens_to_ids(AUDIO).tolist(), audio_llm.audio_end_id]
    assert supervised(ex).tolist() == expected
    # Last prompt token is <|audio_start|>
    first_target = int((ex.labels != IGNORE_INDEX).nonzero()[0])
    assert ex.input_ids[first_target - 1].item() == audio_llm.audio_start_id


def test_end2end_is_self_reconstruction(audio_llm):
    ex = build_example(audio_llm, AUDIO, "hello", "hi", "end2end")
    audio_ids = audio_llm.audio_tokens_to_ids(AUDIO).tolist()
    assert supervised(ex).tolist() == [*audio_ids, audio_llm.audio_end_id]
    assert ex.input_ids.tolist().count(audio_llm.audio_start_id) == 2


def test_audio_ids_never_hit_text_vocab(audio_llm):
    for phase in PHASES:
        ex = build_example(audio_llm, AUDIO, "hello", "hi", phase)
        audio_positions = ex.input_ids >= audio_llm.audio_offset
        assert audio_positions.sum() >= AUDIO.numel()
        assert (ex.input_ids < audio_llm.extended_vocab_size).all()


def test_pad_batch():
    from apollo_voice_engine.training import TrainingExample

    a = TrainingExample(torch.tensor([5, 6, 7]), torch.tensor([IGNORE_INDEX, 6, 7]))
    b = TrainingExample(torch.tensor([8]), torch.tensor([8]))
    batch = pad_batch([a, b], pad_token_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    with pytest.raises(ValueError):
        pad_batch([], 0)


@pytest.mark.parametrize("phase", PHASES)
def test_collate_produces_trainable_batch(audio_llm, phase):
    items = [
        {"audio_tokens": AUDIO, "text": "hello", "language": "hi"},
        {"audio_tokens": AUDIO[:3].repeat(4), "text": "hello world patient", "language": "en"},
    ]
    batch = collate(items, audio_llm=audio_llm, phase=phase)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    loss = audio_llm.model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow through the extended embeddings
    assert audio_llm.model.get_input_embeddings().weight.grad is not None
