"""Run prediction, blending and finalization using the active Python environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


from dotenv import load_dotenv

load_dotenv()


PACKAGE_ROOT = Path(__file__).resolve().parent
RUN_MODULE = "villa_8aa507c.models.run"


def create_parser():
    """Define the command-line options and their defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Extra prediction options can be passed after -- (for example -- --disable_tta). "
            "All prediction parts are run sequentially before blending. "
            "Options can also be set through uppercase environment variables "
            "(for example MODEL_PATH); command-line options take precedence."
        ),
    )
    parser.add_argument(
        "--model_path",
        help="Model path; alternatively set MODEL_PATH",
    )

    parser.add_argument(
        "--input_dir",
        help="Input volume; alternatively set INPUT_DIR",
    )

    parser.add_argument(
        "--output_dir",
        help="Run directory containing logits/, merged_logits.zarr and final_output.zarr",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--num_parts",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Inference batch size; alternatively set BATCH_SIZE",
    )

    parser.add_argument(
        "--patch_size",
        help="Inference tile size Z,Y,X; alternatively set PATCH_SIZE",
    )

    parser.add_argument(
        "--mode",
        choices=("binary", "multiclass"),
        default="binary",
        help="Finalization mode; not passed to inference",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        help="Workers for blending and finalization",
    )

    parser.add_argument(
        "--chunk_size",
        help="Z,Y,X chunk size for blending and finalization",
    )

    parser.add_argument(
        "--sigma_scale",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--threshold_value",
        type=float,
    )

    parser.add_argument(
        "--delete_intermediates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete merged logits after successful finalization; retain patch logits",
    )

    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print commands without executing",
    )

    parser.add_argument(
        "predict_args",
        nargs=argparse.REMAINDER,
    )

    return parser


def environment_arguments(parser, argv):
    """Convert environment values to arguments unless the CLI overrides them."""

    # Leave arguments after -- to inference.py and validate env values with argparse.
    options = argv[: argv.index("--")] if "--" in argv else argv
    supplied = {option.split("=", 1)[0] for option in options}
    env_args = []

    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue

        name = action.dest.upper()
        value = os.environ.get(name)

        if value is None or value == "" or supplied.intersection(action.option_strings):
            continue

        if isinstance(action, argparse.BooleanOptionalAction):
            normalized = value.strip().lower()

            if normalized not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
                parser.error(f"{name} must be one of 1/0, true/false, yes/no, on/off")

            enabled = normalized in ("1", "true", "yes", "on")
            env_args.append(action.option_strings[0 if enabled else 1])
        else:
            env_args.append(f"{action.option_strings[0]}={value}")

    return env_args


def validate_arguments(parser, args):
    """Check required values and options forwarded to inference."""
    missing = [
        f"--{name} or {name.upper()}"
        for name in ("model_path", "input_dir", "output_dir")
        if not getattr(args, name)
    ]

    if missing:
        parser.error("Missing configuration: provide " + ", ".join(missing))

    if args.num_parts < 1:
        parser.error("--num_parts must be at least 1")

    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch_size / BATCH_SIZE must be at least 1")

    if args.patch_size is not None:
        try:
            dimensions = tuple(int(value) for value in args.patch_size.split(","))
        except ValueError:
            parser.error("--patch_size / PATCH_SIZE must contain three positive integers: Z,Y,X")

        if len(dimensions) != 3 or any(value < 1 for value in dimensions):
            parser.error("--patch_size / PATCH_SIZE must contain three positive integers: Z,Y,X")

        args.patch_size = ",".join(map(str, dimensions))

    if args.threshold_value is not None:
        if not args.threshold or args.mode != "binary" or not 0 < args.threshold_value < 1:
            parser.error("--threshold_value requires --threshold, --mode binary and a value in (0, 1)")

    extra = args.predict_args

    if extra[:1] == ["--"]:
        extra = extra[1:]
    elif extra:
        parser.error("Put extra prediction options after --")

    # These options are controlled by the orchestrator so all parts use the same paths.
    reserved = {
        "model_path",
        "input_dir",
        "output_dir",
        "device",
        "num_parts",
        "part_id",
        "mode",
    }

    for option in extra:
        if option.startswith("--") and option[2:].split("=", 1)[0].replace("-", "_") in reserved:
            parser.error(f"{option} is controlled by main.py; pass it before -- (part_id is automatic)")

    return extra


def parse_arguments(argv=None):
    """Read configuration from the environment and command line."""

    parser = create_parser()

    argv = list(sys.argv[1:] if argv is None else argv)

    env_args = environment_arguments(parser, argv)
    args = parser.parse_args(env_args + argv)
    extra = validate_arguments(parser, args)

    return args, extra


def build_commands(argv=None):
    """Build inference, blending and finalization commands for the run."""
    args, extra = parse_arguments(argv)

    root = args.output_dir.rstrip("/\\")
    logits = root + "/logits"
    merged = root + "/merged_logits.zarr"
    final = root + "/final_output.zarr"
    commands = []
    batch_args = ["--batch_size", str(args.batch_size)] if args.batch_size is not None else []
    patch_args = ["--patch_size", args.patch_size] if args.patch_size is not None else []

    for part_id in range(args.num_parts):
        commands.append(
            [
                sys.executable,
                "-m",
                f"{RUN_MODULE}.inference",
                "--model_path",
                args.model_path,
                "--input_dir",
                args.input_dir,
                "--output_dir",
                logits,
                "--device",
                args.device,
                "--num_parts",
                str(args.num_parts),
                "--part_id",
                str(part_id),
                *batch_args,
                *patch_args,
                *extra,
            ]
        )

    common = []
    for name in ("num_workers", "chunk_size"):
        if getattr(args, name) is not None:
            common.extend(["--" + name, str(getattr(args, name))])

    commands.append(
        [
            sys.executable,
            "-m",
            f"{RUN_MODULE}.blending",
            logits,
            merged,
            "--sigma_scale",
            str(args.sigma_scale),
            *common,
        ]
    )

    finalize = [
        sys.executable,
        "-m",
        f"{RUN_MODULE}.finalize_outputs",
        merged,
        final,
        "--mode",
        args.mode,
        *common,
    ]

    if args.threshold:
        finalize.append("--threshold")

    if args.threshold_value is not None:
        finalize.extend(["--threshold_value", str(args.threshold_value)])

    if args.delete_intermediates:
        finalize.append("--delete_intermediates")

    commands.append(finalize)

    return args, commands


def main(argv=None):
    args, commands = build_commands(argv)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PACKAGE_ROOT), env.get("PYTHONPATH"))))

    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {command!r}", flush=True)

        if not args.dry_run:
            try:
                subprocess.run(command, check=True, env=env)
            except subprocess.CalledProcessError as error:
                print(
                    f"Pipeline stopped: subprocess exited with {error.returncode}.",
                    file=sys.stderr,
                )
                return error.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
