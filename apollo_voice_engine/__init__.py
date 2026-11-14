"""
Apollo Omni-Indic Voice Engine.

Speech-to-speech building blocks for conversational AI in Indian languages:
SNAC audio tokenization, an audio-extended Sarvam-1 language model, voice
activity detection, a multilingual safety classifier and demo servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "model_config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the reference model/training/inference configuration YAML."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
