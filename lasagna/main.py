import argparse
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "villa_8aa507c"


def env_bool(value):
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false, yes/no, on/off or 1/0")


def parse_args(argv=None):
    load_dotenv(ROOT.parent / ".env")
    parser = argparse.ArgumentParser(
        description="Run Fiber Trace 3D inference. CLI overrides LASAGNA_* environment variables. "
        "Relative paths are resolved against lasagna/.",
    )
    options = {
        "input": (str, "/path/to/volume.zarr/1"),
        "output": (str, "/path/to/predictions/fiber.lasagna.json"),
        "checkpoint": (str, "models/lasagna-fiber/s1a_128_1_single_8x8_20260801_084232_best91_5k.pt"),
        "devices": (str, "all"),
        "tile-size": (int, 512),
        "overlap": (int, 96),
        "border": (int, 32),
        "inference-scaledown-power": (int, 2),
        "input-reader": (str, "python-zarr"),
        "prefetch-workers": (int, None),
        "prefetch-tiles-per-gpu": (int, None),
        "slots-per-gpu": (int, None),
        "flush-workers": (int, None),
        "accumulator-workers": (int, None),
    }
    for name, (value_type, default) in options.items():
        env_name = "LASAGNA_" + name.upper().replace("-", "_")
        kwargs = {"type": value_type, "default": os.getenv(env_name, default),
                  "help": f"Environment: {env_name} (default: %(default)s)"}
        if name == "input-reader":
            kwargs["choices"] = ("python-zarr", "tensorstore")
        parser.add_argument("--" + name, **kwargs)
    download = parser.add_mutually_exclusive_group()
    download.add_argument("--no-download", dest="no_download", action="store_const", const="true",
                          help="Disable automatic downloads (LASAGNA_NO_DOWNLOAD; default: false)")
    download.add_argument("--download", dest="no_download", action="store_const", const="false",
                          help="Enable automatic downloads")
    parser.set_defaults(no_download=os.getenv("LASAGNA_NO_DOWNLOAD", "false"))
    args = parser.parse_args(argv)
    try:
        args.no_download = env_bool(args.no_download)
    except argparse.ArgumentTypeError as exc:
        parser.error(f"LASAGNA_NO_DOWNLOAD: {exc}")
    return args


def build_command(args):
    command = [sys.executable, "-m", "vesuvius.neural_tracing.fiber_trace_3d.infer"]
    for name, value in vars(args).items():
        if value is None:
            continue
        if name == "no_download":
            if value:
                command.append("--no-download")
            continue
        if name in ("input", "output", "checkpoint"):
            value = os.path.expandvars(os.path.expanduser(value))
        command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def main(argv=None):
    command = build_command(parse_args(argv))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SOURCE)] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
