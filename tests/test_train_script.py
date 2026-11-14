"""Smoke test: run the real training loop on a tiny model and synthetic dataset."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts import train_audio_llm as train


@pytest.fixture
def dataset(tmp_path):
    entries = []
    for i in range(6):
        rel = f"audio/sample_{i}.wav"
        (tmp_path / "audio").mkdir(exist_ok=True)
        t = np.arange(16000) / 16000
        sf.write(
            tmp_path / rel, (0.2 * np.sin(2 * np.pi * (200 + 20 * i) * t)).astype(np.float32), 16000
        )
        entries.append(
            {
                "audio_path": rel,
                "text": "hello world",
                "language": "hi",
                "sampling_rate": 16000,
                "duration": 1.0,
            }
        )
    (tmp_path / "metadata.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("phase", ["understanding", "end2end"])
def test_training_loop_runs_and_checkpoints(
    dataset, tiny_llm_dir, snac_model, tmp_path, monkeypatch, phase
):
    def fake_load(self):
        self.model = snac_model

    monkeypatch.setattr(train.SNACWrapper, "load_model", fake_load)
    config = train.TrainingConfig(
        dataset_path=str(dataset),
        model_name=str(tiny_llm_dir),
        device="cpu",
        batch_size=2,
        gradient_accumulation_steps=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
        languages=["hi"],
    )
    trainer = train.AudioLLMTrainer(config)
    trainer.audio_llm.model.float()
    before = trainer.audio_llm.model.get_input_embeddings().weight.detach().clone()

    trainer.train_phase(phase, num_epochs=1)

    after = trainer.audio_llm.model.get_input_embeddings().weight.detach()
    assert not torch.equal(before, after), "optimizer should update the embeddings"
    assert trainer.global_step >= 1
    best = tmp_path / "ckpt" / f"best_{phase}.pt"
    assert best.exists()

    trainer.global_step = 0
    trainer.load_checkpoint(str(best))  # weights_only=True must accept our checkpoint format
    assert trainer.global_step >= 1
