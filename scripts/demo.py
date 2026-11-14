"""
Command-line demo of the unified SNAC + AudioLLM model.

Modes:
    python scripts/demo.py --check-only                # verify dependencies
    python scripts/demo.py --lang ta --output out.wav  # text prompt -> audio tokens -> WAV
    python scripts/demo.py --input in.wav --output out.wav   # speech -> speech

The base Sarvam-1 checkpoint has never seen audio tokens, so until a
checkpoint trained with ``scripts/train_audio_llm.py`` is supplied via
``--model``, generated audio is not meaningful speech.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("demo")

LANG_ALIASES = {"hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn"}
DEMO_TEXTS = {
    "hi": "नमस्ते, मैं अपोलो वॉयस असिस्टेंट हूँ।",
    "ta": "வணக்கம், நான் அப்பல்லோ குரல் உதவியாளர்.",
    "te": "నమస్కారం, నేను అపోలో వాయిస్ అసిస్టెంట్.",
    "kn": "ನಮಸ್ಕಾರ, ನಾನು ಅಪೊಲೊ ಧ್ವನಿ ಸಹಾಯಕ.",
}
REQUIRED = ["torch", "transformers", "snac", "soundfile", "librosa"]


def check_dependencies() -> bool:
    missing = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            logger.info("ok  %s %s", name, getattr(module, "__version__", ""))
        except ImportError:
            missing.append(name)
    if missing:
        logger.error("Missing dependencies: %s (pip install -e .)", ", ".join(missing))
        return False
    return True


def text_to_speech(model, snac, text: str, max_new_tokens: int):
    """Prompt with text + <|audio_start|> and decode any generated audio tokens."""
    import torch

    prompt = model.text_to_tokens(text)
    start = torch.tensor([[model.audio_start_id]], device=prompt.device)
    input_ids = torch.cat([prompt, start], dim=1)
    output_ids = model.generate_response(
        input_ids=input_ids, max_new_tokens=max_new_tokens, temperature=0.8
    )
    tokens = model.extract_audio_tokens(output_ids[:, input_ids.shape[1] :])
    return None if tokens is None else snac.decode(tokens)


def speech_to_speech(model, snac, audio_path: str, max_new_tokens: int):
    """Encode ``audio_path`` with SNAC, generate, and decode the audio reply."""
    import librosa
    import torch

    waveform, _ = librosa.load(audio_path, sr=snac.sample_rate, mono=True)
    input_tokens = snac.encode(torch.from_numpy(waveform))
    logger.info("Input encoded to %d tokens", input_tokens.shape[-1])

    input_ids = model.build_audio_input_ids(input_tokens)
    output_ids = model.generate_response(input_ids=input_ids, max_new_tokens=max_new_tokens)
    tokens = model.extract_audio_tokens(output_ids[:, input_ids.shape[1] :])
    return None if tokens is None else snac.decode(tokens)


def run(args: argparse.Namespace) -> int:
    if not check_dependencies():
        return 1
    if args.check_only:
        logger.info("All dependencies available.")
        return 0

    from apollo_voice_engine.models import AudioLLM, SNACWrapper

    logger.info("Loading models (run scripts/download_models.py first to pre-fetch)...")
    snac = SNACWrapper()
    snac.load_model()
    model = AudioLLM(model_name=args.model)
    model.load_model()

    if args.input:
        audio = speech_to_speech(model, snac, args.input, args.max_new_tokens)
    else:
        audio = text_to_speech(model, snac, DEMO_TEXTS[args.lang], args.max_new_tokens)

    if audio is None:
        logger.warning("The model produced no audio tokens.")
        return 2

    seconds = audio.shape[-1] / snac.sample_rate
    logger.info("Generated %.2fs of audio", seconds)
    if args.output:
        import soundfile as sf

        sf.write(args.output, audio.detach().float().cpu().numpy().reshape(-1), snac.sample_rate)
        logger.info("Audio saved to %s", args.output)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Apollo Voice Engine CLI demo")
    parser.add_argument(
        "--lang", default="hi", choices=[*DEMO_TEXTS, *LANG_ALIASES], help="Prompt language"
    )
    parser.add_argument("--input", help="Input audio file for speech-to-speech")
    parser.add_argument("--output", help="Where to write the generated WAV")
    parser.add_argument("--model", default="sarvamai/sarvam-1", help="Base or trained checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies")
    args = parser.parse_args()
    args.lang = LANG_ALIASES.get(args.lang, args.lang)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
