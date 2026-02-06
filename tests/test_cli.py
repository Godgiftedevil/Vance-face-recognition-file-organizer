"""Unit tests for the CLI argument parser."""

from pathlib import Path

from face_organizer.cli import build_parser


class TestBuildParser:
    def test_required_input_dir(self):
        parser = build_parser()
        args = parser.parse_args(["my_photos"])
        assert args.input_dir == Path("my_photos")

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["photos"])
        assert args.output_dir == Path("organized")
        assert args.detection_model == "hog"
        assert args.threshold == 0.6
        assert args.action == "copy"
        assert args.no_interactive is False
        assert args.verbose is False

    def test_all_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "input",
            "-o", "output",
            "--detection-model", "cnn",
            "--threshold", "0.4",
            "--action", "move",
            "--no-interactive",
            "--num-jitters", "5",
            "--batch-size", "16",
            "-v",
        ])
        assert args.input_dir == Path("input")
        assert args.output_dir == Path("output")
        assert args.detection_model == "cnn"
        assert args.threshold == 0.4
        assert args.action == "move"
        assert args.no_interactive is True
        assert args.num_jitters == 5
        assert args.batch_size == 16
        assert args.verbose is True
