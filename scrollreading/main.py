import os
import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
for key in ["VOLUME_ZARR", "SURFACE_ZARR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


ROOT = Path(__file__).resolve().parent

SEED_SCRIPT = ROOT / "utils" / "get_seed_points.py"
SIM_PAPER = ROOT / "simpaper10"
BIN2CSV = ROOT / "bin2csv"
CSV2TIFXYZ = ROOT / "csv2tifxyz"

# SEED_SCRIPT = Path(".") / "utils" / "get_seed_points.py"
# SIM_PAPER = Path(".") / "simpaper10"
# BIN2CSV = Path(".") / "bin2csv"
# CSV2TIFXYZ = Path(".") / "csv2tifxyz"


def run_command(command):
    """Run a command and raise an error if it fails."""
    print(f"\n--> Running: {' '.join(map(str, command))}")

    subprocess.run(
        command,
        # cwd=cwd,
        check=True,
    )


def find_seeds(input_tiff, output_dir):
    """Run the seed finder and load the resulting JSON."""

    print("--> Finding seeds...")

    output_filename = "seeds.json"
    
    subprocess.run(
        [
            sys.executable,
            str(SEED_SCRIPT),
            str(input_tiff),
            "--output_path", str(output_dir),
            "--output_filename", output_filename,
        ],
        # cwd=ROOT,
        check=True,
        # capture_output=True,
        text=True,
    )

    with open(output_dir / output_filename, "r") as f:
        data = json.load(f)

    seeds = []

    for d in data["seeds"]:
        seeds.append([d["seed"][2], d["seed"][1], d["seed"][0]])

    # Remove duplicate coordinates while preserving their order.
    unique_seeds = list(dict.fromkeys(tuple(seed) for seed in seeds))

    if not unique_seeds:
        raise RuntimeError("No seeds found. Check the JSON structure.")

    return unique_seeds


def main():
    parser = argparse.ArgumentParser(
        description="Find all seeds and generate patches for each."
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="out",
        help="Base directory for per-seed output",
    )

    args = parser.parse_args()


    input_zarr = Path(os.environ["SURFACE_ZARR"]).resolve()
    if not input_zarr.exists():
        parser.error(f"Input ZARR does not exist: {input_zarr}")

    if not SIM_PAPER.is_file():
        parser.error(f"simpaper10 not found: {SIM_PAPER}")

    try:
        seeds = find_seeds(input_zarr, args.output_dir)
    except Exception as exc:
        sys.exit(f"ERROR while finding seeds: {exc}")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n--> Found {len(seeds)} unique seeds.")

    successful = []
    failed = []

    for index, (x, y, z) in enumerate(seeds, start=1):
        seed_name = f"seed_{index:04d}_{x}_{y}_{z}"
        seed_output = output_root / seed_name
        seed_output.mkdir(parents=True, exist_ok=True)

        print(
            f"\n{'=' * 60}\n"
            f"--> Seed {index}/{len(seeds)}: ({x}, {y}, {z})\n"
            f"--> Output: {seed_output}\n"
            f"{'=' * 60}"
        )

        command = [
            str(SIM_PAPER),
            "g",
            "--output-dir", str(seed_output),
            "--seed", str(x), str(y), str(z),
        ]

        try:
            run_command(command)

            # Find problematic patches
            run_command([
                str(SIM_PAPER),
                "c",
                "--output-dir", str(seed_output),
            ])

            # Generate central 2D coordinates
            run_command([
                str(SIM_PAPER),
                "l",
                "--output-dir", str(seed_output),
            ])

            # Generate visit order and initial 2D placement
            run_command([
                str(SIM_PAPER),
                "v",
                "--output-dir", str(seed_output),
            ])

            # Refine positions using ball-and-spring model
            run_command([
                str(SIM_PAPER),
                "h",
                "--output-dir", str(seed_output),
            ])

            # Flatten the resulting surface
            # run_command([
            #     str(SIM_PAPER),
            #     "f", "30",
            #     "--output-dir", str(seed_output),
            # ])

            # Convert binary patch to CSV
            bin_file = seed_output / "patches" / "patch_0.bin"
            csv_file = seed_output / "patches" / "patch_0.csv"

            print("--> Converting BIN to CSV...")
            with open(csv_file, "w") as f:
                subprocess.run(
                    [str(BIN2CSV), str(bin_file)],
                    stdout=f,
                    check=True,
                )

            # Convert CSV to TIF and XYZ
            print("--> Converting CSV to TIF/XYZ...")
            subprocess.run(
                [str(CSV2TIFXYZ), str(csv_file), str(seed_output / "patch_0")],
                check=True,
            )

            successful.append((x, y, z))
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: Seed ({x}, {y}, {z}) failed with exit code {exc.returncode}.")
            failed.append((x, y, z))

    print("\n--> Processing finished.")
    print(f"Successful: {len(successful)}/{len(seeds)}")
    print(f"Failed:     {len(failed)}/{len(seeds)}")

    if failed:
        print("\nFailed seeds:")
        for seed in failed:
            print(f"  {seed}")

        sys.exit(1)


if __name__ == "__main__":
    main()