"""
Apollo Voice Engine - AudioLLM Training Script

Trains the AudioLLM (Extended Sarvam-1) on speech-to-speech tasks using:
- SNAC for audio tokenization
- IndicVoices dataset for Indian languages

Training Phases:
1. Phase 1: Audio Understanding - Learn to transcribe (audio tokens → text)
2. Phase 2: Audio Generation - Learn to synthesize (text → audio tokens)
3. Phase 3: End-to-End - Audio in, audio out (self-reconstruction for now)

Usage:
    python scripts/train_audio_llm.py --phase understanding --epochs 5
    python scripts/train_audio_llm.py --phase all --epochs 10 --dataset indic_voices_dataset

The dataset directory must contain ``metadata.json`` (see indic_voices_dataset/)
and the referenced audio files.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apollo_voice_engine.models.audio_llm import AudioLLM  # noqa: E402
from apollo_voice_engine.models.snac_wrapper import SNACWrapper  # noqa: E402
from apollo_voice_engine.training import PHASES, collate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_audio_llm")

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Dataset
    dataset_path: str = "indic_voices_dataset"
    metadata_file: str = "metadata.json"

    # Model
    model_name: str = "sarvamai/sarvam-1"
    snac_model: str = "hubertsiuzdak/snac_24khz"

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_epochs: int = 10
    max_audio_length: int = 10  # seconds

    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 500

    # Languages
    languages: list[str] = field(default_factory=lambda: ["hi", "ta", "te", "kn"])


# ============================================================================
# Dataset
# ============================================================================


class IndicVoicesDataset(Dataset):
    """
    Dataset for IndicVoices data.

    Loads audio files and transcriptions for training AudioLLM.
    """

    def __init__(
        self,
        config: TrainingConfig,
        snac: SNACWrapper,
        phase: str = "understanding",  # understanding, generation, end2end
        split: str = "train",
        split_ratio: float = 0.9,
    ):
        self.config = config
        self.snac = snac
        self.phase = phase
        self.split = split

        # Load metadata
        metadata_path = Path(config.dataset_path) / config.metadata_file
        with open(metadata_path, encoding="utf-8") as f:
            self.all_samples = json.load(f)

        # Filter by languages
        self.samples = [s for s in self.all_samples if s["language"] in config.languages]

        # Deterministic train/val split
        random.Random(42).shuffle(self.samples)
        split_idx = int(len(self.samples) * split_ratio)

        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        logger.info(f"Loaded {len(self.samples)} samples for {split} split, phase={phase}")

        # Filter out very short/long samples
        self.samples = [s for s in self.samples if 0.5 <= s["duration"] <= config.max_audio_length]
        logger.info(f"After filtering: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio_path = Path(self.config.dataset_path) / sample["audio_path"]
        audio = self._load_audio(audio_path)

        # Encode audio with SNAC
        with torch.no_grad():
            audio_tokens = self.snac.encode(audio)

        return {
            "audio_tokens": audio_tokens.squeeze(0),
            "text": sample["text"],
            "language": sample["language"],
            "audio_path": str(audio_path),
        }

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load audio resampled to the SNAC sample rate as a ``(1, T)`` tensor."""
        import librosa

        try:
            audio, _ = librosa.load(path, sr=self.snac.sample_rate, mono=True)
        except Exception as exc:
            logger.warning("Error loading audio %s: %s", path, exc)
            # Silent fallback keeps the batch shape valid (1 second)
            return torch.zeros(1, self.snac.sample_rate, device=self.snac.device)
        return torch.from_numpy(audio).float().unsqueeze(0).to(self.snac.device)


# ============================================================================
# Trainer
# ============================================================================


class AudioLLMTrainer:
    """Trainer for the AudioLLM model."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize models
        logger.info("Initializing SNAC...")
        self.snac = SNACWrapper(model_name=config.snac_model, device=config.device)
        self.snac.load_model()

        logger.info("Initializing AudioLLM...")
        self.audio_llm = AudioLLM(
            model_name=config.model_name,
            device=config.device,
            load_in_8bit=False,  # Full precision for training
            load_in_4bit=False,
        )
        self.audio_llm.load_model()

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train_phase(
        self,
        phase: str,
        num_epochs: int,
    ):
        """Train a specific phase."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting Phase: {phase.upper()}")
        logger.info(f"{'=' * 60}\n")

        # Create datasets
        train_dataset = IndicVoicesDataset(self.config, self.snac, phase=phase, split="train")
        val_dataset = IndicVoicesDataset(self.config, self.snac, phase=phase, split="val")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            collate_fn=functools.partial(collate, audio_llm=self.audio_llm, phase=phase),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=functools.partial(collate, audio_llm=self.audio_llm, phase=phase),
        )

        # Optimizer
        optimizer = AdamW(
            self.audio_llm.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * num_epochs // self.config.gradient_accumulation_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, phase)

            # Validate
            val_loss = self._validate(val_loader)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(f"best_{phase}.pt")

            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}_{phase}.pt")

    def _autocast(self):
        enabled = self.config.mixed_precision and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def _train_epoch(self, loader, optimizer, scheduler, phase):
        """Train for one epoch with gradient accumulation."""
        self.audio_llm.model.train()
        total_loss = 0.0
        num_batches = 0
        accum = self.config.gradient_accumulation_steps

        pbar = tqdm(loader, desc=f"Training {phase}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self._autocast():
                outputs = self.audio_llm.model(**batch)
            loss = outputs.loss / accum
            loss.backward()

            is_last = batch_idx + 1 == len(loader)
            if (batch_idx + 1) % accum == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(self.audio_llm.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.global_step += 1
                if self.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}.pt")

            total_loss += loss.item() * accum
            num_batches += 1
            pbar.set_postfix({"loss": total_loss / num_batches, "step": self.global_step})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, loader):
        """Validate the model."""
        self.audio_llm.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc="Validating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self._autocast():
                outputs = self.audio_llm.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

        if num_batches == 0:
            logger.warning("Validation split is empty; reporting inf loss")
            return float("inf")
        return total_loss / num_batches

    def _save_checkpoint(self, filename: str):
        """Save a checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)

        checkpoint = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.audio_llm.model.state_dict(),
            "config": asdict(self.config),
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.audio_llm.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Loaded checkpoint from {path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train AudioLLM on IndicVoices")
    parser.add_argument(
        "--phase",
        type=str,
        default="understanding",
        choices=[*PHASES, "all"],
        help="Training phase",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per phase")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--dataset", type=str, default="indic_voices_dataset", help="Path to dataset"
    )
    parser.add_argument("--model", default="sarvamai/sarvam-1", help="Base model id or path")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Where to save checkpoints")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        dataset_path=args.dataset,
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
    )

    logger.info("Training config: %s", json.dumps(asdict(config), indent=2))

    # Initialize trainer
    trainer = AudioLLMTrainer(config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    if args.phase == "all":
        # Train all phases sequentially
        for phase in PHASES:
            trainer.train_phase(phase, args.epochs)
    else:
        trainer.train_phase(args.phase, args.epochs)

    logger.info("Training complete. Checkpoints saved to %s/", config.checkpoint_dir)


if __name__ == "__main__":
    main()
