"""
Pre-download the models used by the Apollo Voice Engine into the Hugging Face cache.

- ``sarvamai/sarvam-1`` (Indic-optimized 2B LLM, ~5 GB)
- ``hubertsiuzdak/snac_24khz`` (SNAC neural audio codec)

Files are fetched with ``huggingface_hub.snapshot_download`` (no weights are
loaded into memory). Set ``HF_TOKEN`` if a model requires authentication.

Usage:
    python scripts/download_models.py [--cache-dir DIR] [--skip-sarvam] [--skip-snac]
    python scripts/download_models.py --validate-only
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_models")

MODELS = {
    "sarvam": ("sarvamai/sarvam-1", ["*.json", "*.safetensors", "*.model", "*.txt"]),
    "snac": ("hubertsiuzdak/snac_24khz", ["config.json", "pytorch_model.bin"]),
}


def download(repo_id: str, patterns: list[str], cache_dir: str | None, local_only: bool) -> bool:
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=patterns,
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
    except Exception as exc:
        logger.error(
            "%s %s failed: %s", "Validating" if local_only else "Downloading", repo_id, exc
        )
        return False
    logger.info("%s available at %s", repo_id, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Apollo Voice Engine models")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument("--skip-sarvam", action="store_true", help="Skip Sarvam-1")
    parser.add_argument("--skip-snac", action="store_true", help="Skip SNAC")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only check that models are already cached"
    )
    args = parser.parse_args()

    selected = [
        name
        for name in MODELS
        if not (name == "sarvam" and args.skip_sarvam) and not (name == "snac" and args.skip_snac)
    ]
    ok = all(
        download(*MODELS[name], cache_dir=args.cache_dir, local_only=args.validate_only)
        for name in selected
    )
    if not ok:
        logger.error("Some models are missing; see errors above.")
        sys.exit(1)
    logger.info("All selected models are available.")


if __name__ == "__main__":
    main()
