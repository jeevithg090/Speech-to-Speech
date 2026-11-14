"""
Command-line runner for the unified SNAC + AudioLLM engine.

Picks random samples from an IndicVoices-style dataset (``metadata.json`` +
audio files), runs them through :class:`UnifiedVoiceEngine` and writes the
input/output WAVs to ``--output-dir``.

Examples:
    python scripts/unified_cli.py --dataset indic_voices_dataset --language hi
    python scripts/unified_cli.py --max-tokens 50 --num-samples 3
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.unified_demo import UnifiedConfig, UnifiedVoiceEngine  # noqa: E402


def load_random_sample(
    dataset_path: str | Path,
    metadata_file: str = "metadata.json",
    language: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    """
    Return a random metadata entry (with an absolute ``audio_path``), or ``None``.

    ``audio_path`` entries are resolved relative to ``dataset_path``.
    """
    dataset_path = Path(dataset_path)
    metadata_path = dataset_path / metadata_file
    if not metadata_path.exists():
        logging.error("Metadata file not found at %s", metadata_path)
        return None

    entries = json.loads(metadata_path.read_text(encoding="utf-8"))
    if language:
        entries = [e for e in entries if e.get("language") == language]
    if not entries:
        logging.error("No samples found for language %s", language)
        return None

    sample = dict((rng or random).choice(entries))
    audio_path = Path(sample["audio_path"])
    if not audio_path.is_absolute():
        audio_path = dataset_path / audio_path
    sample["audio_path"] = str(audio_path)
    return sample


async def run(args: argparse.Namespace) -> int:
    import librosa
    import soundfile as sf

    config = UnifiedConfig(max_response_tokens=args.max_tokens)
    engine = UnifiedVoiceEngine(config)
    engine.load_models()
    if not engine.models_loaded:
        logging.warning(
            "Models failed to load (%s); results will use fallback TTS", engine.load_error
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for i in range(1, args.num_samples + 1):
        sample = load_random_sample(args.dataset, language=args.language, rng=rng)
        if sample is None:
            return 1
        print(f"\n--- Sample {i}: {sample['audio_path']} ({sample.get('language')})")
        print(f"Reference text: {sample.get('text', '')}")

        try:
            audio, _ = librosa.load(sample["audio_path"], sr=config.sample_rate)
        except Exception as exc:
            logging.error("Could not load audio: %s", exc)
            continue
        sf.write(output_dir / f"sample_{i}_input.wav", audio, config.sample_rate)

        result = await engine.process_speech(
            audio, language=sample.get("language", "hi"), reset_anchor=(i == 1)
        )
        print(json.dumps(result.get("metrics", {}), indent=2))
        print(f"Success: {result['success']}  Fallback: {result.get('fallback_mode')}")

        if result.get("audio"):
            ext = "wav" if result.get("audio_format") == "wav" else "mp3"
            out_path = output_dir / f"sample_{i}_output.{ext}"
            out_path.write_bytes(base64.b64decode(result["audio"]))
            print(f"Saved output audio to {out_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "indic_voices_dataset"))
    parser.add_argument("--language", default="hi", help="Language code filter (hi/ta/te/kn)")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=500, help="Max generated tokens")
    parser.add_argument("--output-dir", default="test_outputs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
