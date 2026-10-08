#!/usr/bin/env python3

import argparse
import numpy as np
import zarr
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def normalize(v):
    """Normalize a vector."""
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("Cannot normalize zero vector.")
    return v / n


def load_surface(path):
    """Load binary Zarr surface volume."""
    z = zarr.open(path, mode="r")

    print(f"Surface shape: {z.shape}")
    print(f"Surface dtype: {z.dtype}")

    surface = np.asarray(z) > 0

    print(f"Surface voxels: {surface.sum():,}")

    return surface


def extract_local_surface(surface, seed_zyx, radius):
    """
    Extract surface voxels in a cubic region around the seed.

    Input/output coordinates:
        volume: Z,Y,X
        returned points: X,Y,Z
    """

    z, y, x = seed_zyx

    z0 = max(0, int(np.floor(z - radius)))
    z1 = min(surface.shape[0], int(np.ceil(z + radius + 1)))

    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(surface.shape[1], int(np.ceil(y + radius + 1)))

    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(surface.shape[2], int(np.ceil(x + radius + 1)))

    local = surface[z0:z1, y0:y1, x0:x1]

    points = np.argwhere(local)

    points[:, 0] += z0
    points[:, 1] += y0
    points[:, 2] += x0

    # Z,Y,X -> X,Y,Z
    points_xyz = points[:, [2, 1, 0]].astype(np.float64)

    return points_xyz


def find_local_tangent_plane(points, seed):
    """
    Estimate the local tangent plane using PCA.
    """

    centered = points - seed

    covariance = centered.T @ centered / len(centered)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Eigenvector belonging to smallest eigenvalue = surface normal
    normal = eigenvectors[:, 0]

    # Two largest eigenvectors = tangent directions
    axis_a = eigenvectors[:, 2]
    axis_b = eigenvectors[:, 1]

    normal = normalize(normal)
    axis_a = normalize(axis_a)
    axis_b = normalize(axis_b)

    return normal, axis_a, axis_b


def find_axes(points, seed, normal):
    """
    Find two local directions in the tangent plane.

    We search for two surface points whose projected directions
    are as close to orthogonal as possible.
    """

    vectors = points - seed

    # Project vectors into tangent plane
    vectors -= np.outer(vectors @ normal, normal)

    lengths = np.linalg.norm(vectors, axis=1)

    valid = lengths > 1.0

    vectors = vectors[valid]
    points = points[valid]
    lengths = lengths[valid]

    if len(vectors) < 2:
        raise RuntimeError(
            "Not enough local surface points to determine axes."
        )

    directions = vectors / lengths[:, None]

    # Ignore very distant points. We want local directions.
    max_distance = np.percentile(lengths, 30)

    local = lengths <= max_distance

    directions = directions[local]
    points = points[local]

    best_dot = float("inf")
    best_i = None
    best_j = None

    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            dot = abs(np.dot(directions[i], directions[j]))

            if dot < best_dot:
                best_dot = dot
                best_i = i
                best_j = j

    if best_i is None:
        raise RuntimeError("Could not find two local directions.")

    axis1 = directions[best_i]
    axis2 = directions[best_j]

    # Make axes point in a deterministic direction.
    # Pipeline9 itself does not enforce this orientation.
    for axis in (axis1, axis2):
        pass

    return axis1, axis2, best_dot


def main():

    parser = argparse.ArgumentParser(
        description="Determine Pipeline9-style seed axes from a binary surface Zarr."
    )

    parser.add_argument(
        "--surface",
        required=True,
        help="Binary surface Zarr volume (Z,Y,X).",
    )

    parser.add_argument(
        "--seed",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Seed coordinate in X Y Z order.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=20.0,
        help="Radius of local surface region in voxels.",
    )

    args = parser.parse_args()

    seed = np.asarray(args.seed, dtype=np.float64)

    print()
    print("Loading surface...")
    surface = load_surface(args.surface)

    # Surface uses Z,Y,X
    seed_zyx = seed[[2, 1, 0]]

    print()
    print("Seed:")
    print(f"  X = {seed[0]}")
    print(f"  Y = {seed[1]}")
    print(f"  Z = {seed[2]}")

    print()
    print(f"Extracting local surface (radius={args.radius})...")

    points = extract_local_surface(
        surface,
        seed_zyx,
        args.radius,
    )

    print(f"Local surface points: {len(points):,}")

    if len(points) < 10:
        raise RuntimeError(
            "Too few surface points around the seed."
        )

    print()
    print("Estimating local tangent plane...")

    normal, tangent_a, tangent_b = find_local_tangent_plane(
        points,
        seed,
    )

    print("Surface normal:")
    print(" ", normal)

    print()
    print("Finding two local axes...")

    axis1, axis2, dot = find_axes(
        points,
        seed,
        normal,
    )

    print()
    print("=" * 60)
    print("Pipeline9 seed")
    print("=" * 60)

    print()
    print("Seed:")
    print(f"SEED_X={seed[0]:.9f}")
    print(f"SEED_Y={seed[1]:.9f}")
    print(f"SEED_Z={seed[2]:.9f}")

    print()
    print("Axis 1:")
    print(f"SEED_AXIS1_X={axis1[0]:.9f}")
    print(f"SEED_AXIS1_Y={axis1[1]:.9f}")
    print(f"SEED_AXIS1_Z={axis1[2]:.9f}")

    print()
    print("Axis 2:")
    print(f"SEED_AXIS2_X={axis2[0]:.9f}")
    print(f"SEED_AXIS2_Y={axis2[1]:.9f}")
    print(f"SEED_AXIS2_Z={axis2[2]:.9f}")

    print()
    print("Axis dot product:")
    print(f"  {np.dot(axis1, axis2):.9f}")

    print()
    print("Surface normal:")
    print(
        f"  {normal[0]:.9f} "
        f"{normal[1]:.9f} "
        f"{normal[2]:.9f}"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()