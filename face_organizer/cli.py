"""Command-line interface for Face Organizer."""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="face_organizer",
        description=(
            "Organize a personal photo collection by detected faces — "
            "fully offline, no cloud APIs."
        ),
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the photos to organize.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("organized"),
        help="Directory where per-person folders will be created (default: organized/).",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model: hog (fast/CPU) or cnn (accurate/GPU). Default: hog.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Clustering distance threshold (0–1). Lower = stricter. Default: 0.6.",
    )
    parser.add_argument(
        "--action",
        choices=["copy", "move"],
        default="copy",
        help="Copy or move photos into person folders. Default: copy.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive labeling; use auto-generated labels.",
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=1,
        help="Number of encoding re-samples (higher = slower but more accurate). Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing. Default: 32.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser


def main(argv: list | None = None) -> int:
    """Entry-point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        detection_model=args.detection_model,
        cluster_threshold=args.threshold,
        file_action=args.action,
        encoding_num_jitters=args.num_jitters,
        batch_size=args.batch_size,
    )

    try:
        config.validate()
    except ValueError as exc:
        logging.error("Invalid configuration: %s", exc)
        return 1

    if not config.input_dir.is_dir():
        logging.error("Input directory does not exist: %s", config.input_dir)
        return 1

    result = run_pipeline(config, interactive=not args.no_interactive)
    if result:
        print(f"\n✓ Organized photos into {len(result)} group(s):")
        for label, directory in sorted(result.items()):
            print(f"  • {label}: {directory}")
    else:
        print("\nNo faces found — nothing to organize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
