"""Shared fixtures: tiny, offline stand-ins for SNAC and Sarvam-1."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apollo_voice_engine.models.audio_llm import AudioLLM  # noqa: E402
from apollo_voice_engine.models.snac_wrapper import SNACWrapper  # noqa: E402

TINY_WORDS = [
    "hello", "world", "transcribe", "the", "following", "audio", "in", "hi", "en",
    "synthesize", "text", "patient", "pain", ":", ".",
]  # fmt: skip


@pytest.fixture(scope="session")
def snac_model():
    """Randomly initialised SNAC with the real 24 kHz architecture (no download)."""
    from snac import SNAC

    torch.manual_seed(0)
    model = SNAC(
        sampling_rate=24000,
        encoder_dim=48,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=64,  # real model uses 1024; smaller decoder keeps tests fast
        decoder_rates=[8, 8, 4, 2],
        attn_window_size=None,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[4, 2, 1],
    )
    return model.eval()


@pytest.fixture
def snac(snac_model) -> SNACWrapper:
    wrapper = SNACWrapper(device="cpu")
    wrapper.model = snac_model
    return wrapper


@pytest.fixture(scope="session")
def tiny_llm_dir(tmp_path_factory) -> Path:
    """A tiny Llama checkpoint + word-level tokenizer saved to disk."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny_llm")
    vocab = {tok: i for i, tok in enumerate(["<unk>", "<pad>", "<s>", "</s>", *TINY_WORDS])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(path)

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return path


@pytest.fixture
def audio_llm(tiny_llm_dir) -> AudioLLM:
    model = AudioLLM(model_name=str(tiny_llm_dir), device="cpu")
    model.load_model()
    model.model.float()  # bf16 is slow/unsupported for some CPU ops
    model.speaker_proj.float()
    return model
