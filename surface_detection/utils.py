# This code was adapted from the following Kaggle notebook:
# https://www.kaggle.com/code/mariusheuser/local-interpolation-interference
#
# The original code is licensed under the Apache License, Version 2.0.
# The source cell from the original notebook is indicated above each
# corresponding code section below.





# Source: Kaggle notebook, Cell 1
# https://www.kaggle.com/code/mariusheuser/local-interpolation-interference


import scipy
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, binary_erosion, distance_transform_edt, binary_closing, generate_binary_structure
import scipy.ndimage as ndi
from scipy.interpolate import griddata
from skimage.measure import label
from skimage.morphology import ball
from scipy.ndimage import median_filter
from collections import deque
from scipy.spatial import cKDTree
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
from skimage.measure import euler_number

# ============================================================================
# NUMBA-OPTIMIZED RASTERIZATION (10-50x faster)
# ============================================================================

@jit(nopython=True, fastmath=True)
def rasterize_triangle_numba(p1, p2, p3, volume):
    """Numba-optimized triangle rasterization. Expected speedup: 10-50x."""
    min_z = max(0, int(np.floor(min(p1[0], p2[0], p3[0]))))
    max_z = min(volume.shape[0] - 1, int(np.ceil(max(p1[0], p2[0], p3[0]))))
    min_y = max(0, int(np.floor(min(p1[1], p2[1], p3[1]))))
    max_y = min(volume.shape[1] - 1, int(np.ceil(max(p1[1], p2[1], p3[1]))))
    min_x = max(0, int(np.floor(min(p1[2], p2[2], p3[2]))))
    max_x = min(volume.shape[2] - 1, int(np.ceil(max(p1[2], p2[2], p3[2]))))

    v0 = p2 - p1
    v1 = p3 - p1
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    denom = d00 * d11 - d01 * d01

    if abs(denom) < 1e-10:
        return

    inv_denom = 1.0 / denom

    for z in range(min_z, max_z + 1):
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                v2_0 = z - p1[0]
                v2_1 = y - p1[1]
                v2_2 = x - p1[2]
                d20 = v2_0 * v0[0] + v2_1 * v0[1] + v2_2 * v0[2]
                d21 = v2_0 * v1[0] + v2_1 * v1[1] + v2_2 * v1[2]
                v = (d11 * d20 - d01 * d21) * inv_denom
                w = (d00 * d21 - d01 * d20) * inv_denom
                u = 1.0 - v - w
                if u >= -0.01 and v >= -0.01 and w >= -0.01:
                    volume[z, y, x] = True


@jit(nopython=True, fastmath=True, parallel=True)
def rasterize_surface_numba(grid_points, volume, samples_per_edge=5):
    """
    Numba-optimized surface rasterization with parallel processing.
    Expected speedup: 20-100x faster than pure Python.
    """
    grid_resolution = grid_points.shape[0]

    for i in prange(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            p1 = grid_points[i, j]
            p2 = grid_points[i+1, j]
            p3 = grid_points[i, j+1]
            p4 = grid_points[i+1, j+1]

            if (np.isnan(p1[0]) or np.isnan(p2[0]) or
                np.isnan(p3[0]) or np.isnan(p4[0])):
                continue

            for u_idx in range(samples_per_edge):
                u = u_idx / (samples_per_edge - 1) if samples_per_edge > 1 else 0.5
                for v_idx in range(samples_per_edge):
                    v = v_idx / (samples_per_edge - 1) if samples_per_edge > 1 else 0.5

                    point_0 = ((1-u)*(1-v)*p1[0] + u*(1-v)*p2[0] +
                              (1-u)*v*p3[0] + u*v*p4[0])
                    point_1 = ((1-u)*(1-v)*p1[1] + u*(1-v)*p2[1] +
                              (1-u)*v*p3[1] + u*v*p4[1])
                    point_2 = ((1-u)*(1-v)*p1[2] + u*(1-v)*p2[2] +
                              (1-u)*v*p3[2] + u*v*p4[2])

                    iz = int(np.round(point_0))
                    iy = int(np.round(point_1))
                    ix = int(np.round(point_2))

                    if (0 <= iz < volume.shape[0] and
                        0 <= iy < volume.shape[1] and
                        0 <= ix < volume.shape[2]):
                        volume[iz, iy, ix] = True

# ============================================================================
# ALGORITHMIC OPTIMIZATIONS
# ============================================================================

def adaptive_grid_resolution(component, base_resolution=100, max_resolution=150):
    """Dynamically adjust grid resolution based on component size."""
    num_voxels = np.sum(component)
    if num_voxels < 500:
        return min(30, base_resolution)
    elif num_voxels < 2000:
        return min(50, base_resolution)
    elif num_voxels < 5000:
        return min(70, base_resolution)
    elif num_voxels < 15000:
        return base_resolution
    else:
        return min(max_resolution, base_resolution + 20)


def should_skip_smoothing(component, coverage_threshold=0.8):
    """Determine if a component needs smoothing based on planarity."""
    coords = np.column_stack(np.nonzero(component))
    coords_mean = coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(coords - coords_mean, full_matrices=False)
    if S[2] / S[0] < 0.05:
        return True
    return False


def zero_volume_faces(volume, thickness=5):
    """Optimized face zeroing using slicing."""
    result = volume.copy()
    result[:thickness, :, :] = False
    result[-thickness:, :, :] = False
    result[:, :thickness, :] = False
    result[:, -thickness:, :] = False
    result[:, :, :thickness] = False
    result[:, :, -thickness:] = False
    return result


# ============================================================================
# OPTIMIZED MAIN FITTING FUNCTION
# ============================================================================

def fit_curved_sheet_to_component_optimized(
    component,
    grid_resolution=100,
    thickness=3,
    smoothing=1.0,
    use_median_filter=True,
    max_distance=10,
    use_numba=True,
    adaptive_resolution=True,
    samples_per_edge=8
):
    """
    OPTIMIZED version of fit_curved_sheet_to_component.
    Key optimizations:
    1. Numba JIT compilation for rasterization (10-50x speedup)
    2. Adaptive grid resolution (2-4x speedup for small components)
    3. Skip smoothing when not needed
    """
    coords = np.column_stack(np.nonzero(component))
    if len(coords) < 10:
        return component.copy()

    if adaptive_resolution:
        grid_resolution = adaptive_grid_resolution(component, grid_resolution)
        print(f"    Using adaptive grid resolution: {grid_resolution}")

    coords_mean = coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(coords - coords_mean, full_matrices=False)
    tangent1, tangent2 = Vt[0], Vt[1]
    normal_guess = Vt[2]

    uv_coords = (coords - coords_mean) @ np.column_stack([tangent1, tangent2])
    w_coords = (coords - coords_mean) @ normal_guess

    if len(coords) > 5000:
        indices = np.random.choice(len(coords), 5000, replace=False)
        uv_coords_sample = uv_coords[indices]
        w_coords_sample = w_coords[indices]
    else:
        uv_coords_sample = uv_coords
        w_coords_sample = w_coords

    u_min, u_max = uv_coords[:,0].min(), uv_coords[:,0].max()
    v_min, v_max = uv_coords[:,1].min(), uv_coords[:,1].max()
    u_padding = (u_max - u_min) * 0.05
    v_padding = (v_max - v_min) * 0.05

    grid_u, grid_v = np.meshgrid(
        np.linspace(u_min - u_padding, u_max + u_padding, num=grid_resolution),
        np.linspace(v_min - v_padding, v_max + v_padding, num=grid_resolution),
        indexing='ij'
    )

    try:
        w_grid = griddata(uv_coords_sample, w_coords_sample, (grid_u, grid_v), method='linear')
    except:
        w_grid = griddata(uv_coords_sample, w_coords_sample, (grid_u, grid_v), method='nearest')

    if np.any(np.isnan(w_grid)):
        mask = np.isnan(w_grid)
        w_grid_nearest = griddata(uv_coords_sample, w_coords_sample, (grid_u, grid_v), method='nearest')
        w_grid[mask] = w_grid_nearest[mask]

    if use_median_filter:
        w_grid = median_filter(w_grid, size=3)

    skip_smooth = should_skip_smoothing(component)
    
    if smoothing > 0 and not skip_smooth:
        w_grid = gaussian_filter(w_grid, sigma=smoothing)
    elif skip_smooth:
        print(f"    Skipping smoothing (component already planar)")

    # grid_padding = 0.02
    # if grid_padding is not None:
    #     u_data_min, u_data_max = uv_coords[:,0].min(), uv_coords[:,0].max()
    #     v_data_min, v_data_max = uv_coords[:,1].min(), uv_coords[:,1].max()
    #     u_range = u_data_max - u_data_min
    #     v_range = v_data_max - v_data_min
    #     u_pad = u_range * grid_padding
    #     v_pad = v_range * grid_padding
    #     grid_mask = ((grid_u >= u_data_min - u_pad) & (grid_u <= u_data_max + u_pad) &
    #                  (grid_v >= v_data_min - v_pad) & (grid_v <= v_data_max + v_pad))
    #     w_grid[~grid_mask] = np.nan

    #Grid trimming with KDTree (already optimized in original)
    tree = cKDTree(uv_coords)
    threshold = (u_max - u_min + v_max - v_min) / (2 * grid_resolution) * 2
    
    grid_uv_flat = np.column_stack([grid_u.ravel(), grid_v.ravel()])
    distances, _ = tree.query(grid_uv_flat, k=1)
    distances = distances.reshape(grid_resolution, grid_resolution)
    
    original_data_mask = distances <= threshold
    
    # Flood-fill from edges
    grid_mask = np.ones_like(w_grid, dtype=bool)
    visited = np.zeros_like(w_grid, dtype=bool)
    queue = deque()
    
    for i in range(grid_resolution):
        queue.append((i, 0))
        queue.append((i, grid_resolution - 1))
        visited[i, 0] = True
        visited[i, grid_resolution - 1] = True
    
    for j in range(1, grid_resolution - 1):
        queue.append((0, j))
        queue.append((grid_resolution - 1, j))
        visited[0, j] = True
        visited[grid_resolution - 1, j] = True
    
    while queue:
        i, j = queue.popleft()
        
        if not original_data_mask[i, j]:
            grid_mask[i, j] = False
            
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < grid_resolution and 
                    0 <= nj < grid_resolution and 
                    not visited[ni, nj]):
                    visited[ni, nj] = True
                    queue.append((ni, nj))
    
    w_grid[~grid_mask] = np.nan
    
    grid_points = (coords_mean +
                   grid_u[...,None] * tangent1 +
                   grid_v[...,None] * tangent2 +
                   w_grid[...,None] * normal_guess)

    Z, Y, X = component.shape
    sheet_volume = np.zeros_like(component, dtype=bool)

    if use_numba:
        rasterize_surface_numba(grid_points, sheet_volume, samples_per_edge=samples_per_edge)
    else:
        rasterize_surface_dense_sampling_original(grid_points, sheet_volume, samples_per_quad=samples_per_edge)

    sheet_volume = zero_volume_faces(sheet_volume, thickness=5)

    if thickness > 0:
        iterations = max(1, thickness // 2)
        struct_elem = np.array([
            [[0,1,0], [1,1,1], [0,1,0]],
            [[1,1,1], [1,1,1], [1,1,1]],
            [[0,1,0], [1,1,1], [0,1,0]]
        ], dtype=bool)
        sheet_volume = binary_dilation(sheet_volume, structure=struct_elem, iterations=iterations)

    for z in range(Z):
        if np.any(sheet_volume[z]):
            sheet_volume[z] = binary_fill_holes(sheet_volume[z])

    # struct = ndi.generate_binary_structure(3, 3)
    # sheet_volume = ndi.binary_closing(sheet_volume, structure=struct, iterations=1)

    return sheet_volume


def rasterize_surface_dense_sampling_original(grid_points, volume, samples_per_quad=5):
    """Original Python implementation for fallback."""
    grid_resolution = grid_points.shape[0]
    for i in range(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            p1 = grid_points[i, j]
            p2 = grid_points[i+1, j]
            p3 = grid_points[i, j+1]
            p4 = grid_points[i+1, j+1]
            if (np.isnan(p1).any() or np.isnan(p2).any() or
                np.isnan(p3).any() or np.isnan(p4).any()):
                continue
            for u in np.linspace(0, 1, samples_per_quad):
                for v in np.linspace(0, 1, samples_per_quad):
                    point = ((1-u)*(1-v)*p1 + u*(1-v)*p2 + (1-u)*v*p3 + u*v*p4)
                    point = point.round().astype(int)
                    if (0 <= point[0] < volume.shape[0] and
                        0 <= point[1] < volume.shape[1] and
                        0 <= point[2] < volume.shape[2]):
                        volume[point[0], point[1], point[2]] = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_dice_score(mask1, mask2):
    """Calculate Dice coefficient between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return 0.0
    return 2.0 * intersection / sum_masks


def calculate_coverage_score(original, fitted):
    """Calculate how well the fitted sheet covers the original positive pixels."""
    original_pixels = np.sum(original)
    if original_pixels == 0:
        return 0.0
    return np.sum(original & fitted) / original_pixels





# Source: Kaggle notebook, Cell 2
# https://www.kaggle.com/code/mariusheuser/local-interpolation-interference


import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, binary_erosion, distance_transform_edt, binary_closing, generate_binary_structure
import scipy.ndimage as ndi
from scipy.interpolate import griddata
from skimage.measure import label
from skimage.morphology import ball
from scipy.ndimage import median_filter
from collections import deque
from scipy.spatial import cKDTree
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
from skimage.measure import euler_number

# [Keep all the NUMBA functions and other helpers from the original code]
# ... (rasterize_triangle_numba, rasterize_surface_numba, etc.) ...

# ============================================================================
# PATCH-WISE (CUBE-BASED) TOPOLOGY CHECKING AND INTERPOLATION
# ============================================================================

def divide_component_into_cubes(component, cube_size=32, overlap=8):
    """
    Divide a component into smaller cubes with optional overlap.
    
    Parameters:
    -----------
    component : ndarray (bool)
        Binary mask of the component
    cube_size : int
        Size of each cube dimension
    overlap : int
        Overlap between adjacent cubes (helps with boundary artifacts)
    
    Returns:
    --------
    list of dict
        Each dict contains:
        - 'coords': (z_start, z_end, y_start, y_end, x_start, x_end)
        - 'center': (z_center, y_center, x_center)
        - 'mask': boolean cube region from component
        - 'at_volume_boundary': dict indicating which faces touch volume boundaries
    """
    shape = component.shape
    cubes = []
    
    # Calculate stride (cube_size - overlap ensures overlap between cubes)
    stride = cube_size - overlap
    
    for z in range(0, shape[0], stride):
        z_end = min(z + cube_size, shape[0])
        if z_end - z < cube_size // 2:  # Skip tiny remainder cubes
            continue
            
        for y in range(0, shape[1], stride):
            y_end = min(y + cube_size, shape[1])
            if y_end - y < cube_size // 2:
                continue
                
            for x in range(0, shape[2], stride):
                x_end = min(x + cube_size, shape[2])
                if x_end - x < cube_size // 2:
                    continue
                
                # Extract cube region
                cube_mask = component[z:z_end, y:y_end, x:x_end]
                
                # Only include cubes that contain component voxels
                if np.sum(cube_mask) > 10:  # Minimum voxel threshold
                    # Detect which faces are at volume boundaries
                    at_volume_boundary = {
                        'z_min': (z == 0),
                        'z_max': (z_end == shape[0]),
                        'y_min': (y == 0),
                        'y_max': (y_end == shape[1]),
                        'x_min': (x == 0),
                        'x_max': (x_end == shape[2])
                    }
                    
                    cubes.append({
                        'coords': (z, z_end, y, y_end, x, x_end),
                        'center': (
                            (z + z_end) // 2,
                            (y + y_end) // 2,
                            (x + x_end) // 2
                        ),
                        'mask': cube_mask,
                        'shape': cube_mask.shape,
                        'at_volume_boundary': at_volume_boundary
                    })
    
    return cubes


def check_cube_topology(cube_mask, min_voxels=20):
    """
    Check if a cube region has topological issues (holes).
    
    Parameters:
    -----------
    cube_mask : ndarray (bool)
        Binary mask of cube region
    min_voxels : int
        Minimum voxels required to perform check
    
    Returns:
    --------
    tuple (has_hole, beta1, chi)
        has_hole: True if beta1 > 0
        beta1: First Betti number (number of holes)
        chi: Euler characteristic
    """
    if np.sum(cube_mask) < min_voxels:
        return False, 0, 1
    
    try:
        chi = euler_number(cube_mask.astype(int), connectivity=1)
        beta1 = 1 - chi
        has_hole = beta1 > 0
        return has_hole, beta1, chi
    except:
        return False, 0, 1


def find_overlapping_cube_groups(cubes_with_holes):
    """
    Group cubes that overlap or are adjacent into clusters.
    Uses connected components on a cube adjacency graph.
    
    Parameters:
    -----------
    cubes_with_holes : list of dict
        List of cube dictionaries that have holes
    
    Returns:
    --------
    list of list
        Each inner list contains indices of cubes in a connected group
    """
    n = len(cubes_with_holes)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    
    # Build adjacency matrix
    adjacent = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            if cubes_overlap_or_adjacent(
                cubes_with_holes[i]['coords'],
                cubes_with_holes[j]['coords'],
                adjacency_distance=2  # Consider adjacent if within 2 voxels
            ):
                adjacent[i, j] = True
                adjacent[j, i] = True
    
    # Find connected components in adjacency graph
    visited = np.zeros(n, dtype=bool)
    groups = []
    
    for i in range(n):
        if not visited[i]:
            # BFS to find connected component
            group = []
            queue = [i]
            visited[i] = True
            
            while queue:
                idx = queue.pop(0)
                group.append(idx)
                
                for j in range(n):
                    if adjacent[idx, j] and not visited[j]:
                        visited[j] = True
                        queue.append(j)
            
            groups.append(group)
    
    return groups


def cubes_overlap_or_adjacent(coords1, coords2, adjacency_distance=2):
    """
    Check if two cubes overlap or are adjacent (within distance threshold).
    
    Parameters:
    -----------
    coords1, coords2 : tuple
        (z_start, z_end, y_start, y_end, x_start, x_end)
    adjacency_distance : int
        Maximum gap distance to consider cubes as adjacent
    
    Returns:
    --------
    bool
        True if cubes overlap or are adjacent
    """
    z1_s, z1_e, y1_s, y1_e, x1_s, x1_e = coords1
    z2_s, z2_e, y2_s, y2_e, x2_s, x2_e = coords2
    
    # Check for overlap or adjacency in each dimension
    z_overlap = not (z1_e + adjacency_distance < z2_s or z2_e + adjacency_distance < z1_s)
    y_overlap = not (y1_e + adjacency_distance < y2_s or y2_e + adjacency_distance < y1_s)
    x_overlap = not (x1_e + adjacency_distance < x2_s or x2_e + adjacency_distance < x2_s)
    
    return z_overlap and y_overlap and x_overlap


def merge_cube_regions(cubes, cube_indices, component_shape):
    """
    Merge multiple cubes into a single bounding region.
    Also merges the volume boundary information.
    
    Returns:
    --------
    tuple
        (coords, at_volume_boundary) where:
        - coords: (z_start, z_end, y_start, y_end, x_start, x_end) of merged region
        - at_volume_boundary: dict of boundary flags for merged region
    """
    z_min = component_shape[0]
    z_max = 0
    y_min = component_shape[1]
    y_max = 0
    x_min = component_shape[2]
    x_max = 0
    
    # Merge boundaries - if ANY cube touches a volume boundary, the merged region does
    at_volume_boundary = {
        'z_min': False, 'z_max': False,
        'y_min': False, 'y_max': False,
        'x_min': False, 'x_max': False
    }
    
    for idx in cube_indices:
        z_s, z_e, y_s, y_e, x_s, x_e = cubes[idx]['coords']
        z_min = min(z_min, z_s)
        z_max = max(z_max, z_e)
        y_min = min(y_min, y_s)
        y_max = max(y_max, y_e)
        x_min = min(x_min, x_s)
        x_max = max(x_max, x_e)
        
        # Merge boundary flags
        for key in at_volume_boundary:
            at_volume_boundary[key] |= cubes[idx]['at_volume_boundary'][key]
    
    coords = (z_min, z_max, y_min, y_max, x_min, x_max)
    return coords, at_volume_boundary





# Source: Kaggle notebook, Cell 3
# https://www.kaggle.com/code/mariusheuser/local-interpolation-interference


def interpolate_cube_region_with_quality_check(
    component,
    cube_coords,
    at_volume_boundary,
    alternative_volumes=None,
    border_thickness=4,
    grid_resolution=80,
    thickness=3,
    smoothing=2.0,
    max_distance=10,
    samples_per_edge=8,
    min_dice=0.65,
    min_coverage=0.65,
    alt_min_dice=0.60,
    alt_min_coverage=0.60,
    min_component_size=100
):
    """
    Interpolate a cube region with quality checking and alternative volumes.
    
    Strategy:
    1. Extract and interpolate cube region
    2. Check dice/coverage scores
    3. If scores too low, try alternative volumes (pre-segmented at different thresholds)
    4. Only replace interior voxels, but skip border preservation at volume edges
    
    Parameters:
    -----------
    component : ndarray (bool)
        Full component mask
    cube_coords : tuple
        (z_start, z_end, y_start, y_end, x_start, x_end)
    at_volume_boundary : dict
        Flags indicating which cube faces touch volume boundaries
    alternative_volumes : list of ndarray (bool), optional
        Pre-computed segmentations at different thresholds to try if quality is low
    border_thickness : int
        Thickness of border to preserve (except at volume boundaries)
    min_dice : float
        Minimum Dice score required for main interpolation
    min_coverage : float
        Minimum coverage score required for main interpolation
    alt_min_dice : float
        Minimum Dice score required for alternative volumes
    alt_min_coverage : float
        Minimum coverage score required for alternative volumes
    min_component_size : int
        Minimum voxels for components to keep (default: 100)
    
    Returns:
    --------
    tuple (result_component, success, metrics)
        result_component: Modified component
        success: True if interpolation met quality thresholds
        metrics: dict with dice, coverage, etc.
    """
    z_s, z_e, y_s, y_e, x_s, x_e = cube_coords
    
    # Extract cube region
    cube_mask = component[z_s:z_e, y_s:y_e, x_s:x_e].copy()
    original_cube = cube_mask.copy()
    
    if np.sum(cube_mask) < 20:
        return component, False, {'dice': 0.0, 'coverage': 0.0, 'reason': 'too_small'}
    
    print(f"    Interpolating cube region: "
          f"z[{z_s}:{z_e}], y[{y_s}:{y_e}], x[{x_s}:{x_e}]")
    
    # ========================================================================
    # ATTEMPT 1: Try original mask
    # ========================================================================
    print(f"      Attempt 1: Original mask")
    
    try:
        interpolated_cube = fit_curved_sheet_to_component_optimized(
            cube_mask,
            grid_resolution=grid_resolution,
            thickness=thickness,
            smoothing=smoothing,
            max_distance=max_distance,
            use_numba=True,
            adaptive_resolution=True,
            samples_per_edge=samples_per_edge
        )
    except Exception as e:
        print(f"      Failed to interpolate: {e}")
        interpolated_cube = None
    
    if interpolated_cube is not None:
        dice = calculate_dice_score(original_cube, interpolated_cube)
        coverage = calculate_coverage_score(original_cube, interpolated_cube)
        print(f"      Dice={dice:.3f}, Coverage={coverage:.3f}")
        
        if dice >= min_dice and coverage >= min_coverage:
            print(f"      ✓ Quality threshold met!")
            best_result = interpolated_cube
            best_metrics = {'dice': dice, 'coverage': coverage, 'attempt': 0, 'num_parts': 1}
            # Skip to replacement section
            success = True
            interpolated_cube = best_result
            final_dice = best_metrics['dice']
            final_coverage = best_metrics['coverage']
            
            # Jump to interior mask creation
            cube_shape = cube_mask.shape
            interior_mask = np.ones(cube_shape, dtype=bool)
            
            # ... (rest of interior mask code below)
            # For brevity, let me continue from here
        else:
            print(f"      ✗ Quality too low, trying alternative volumes...")
            best_result = interpolated_cube
            best_metrics = {'dice': dice, 'coverage': coverage, 'attempt': 0, 'num_parts': 1}
    else:
        best_result = None
        best_metrics = {'dice': 0.0, 'coverage': 0.0}
    
    # ========================================================================
    # ATTEMPT 2+: Try alternative volumes (if quality was low)
    # ========================================================================
    if best_metrics['dice'] < min_dice or best_metrics['coverage'] < min_coverage:
        if alternative_volumes is None or len(alternative_volumes) == 0:
            print(f"      No alternative volumes available")
            if best_result is None:
                return component, False, {'dice': 0.0, 'coverage': 0.0, 'reason': 'no_alternatives'}
        else:
            print(f"      Trying {len(alternative_volumes)} alternative volume(s)...")
            
            for alt_idx, alt_volume in enumerate(alternative_volumes):
                print(f"      Attempt {alt_idx + 2}: Alternative volume {alt_idx + 1}")
                
                # Extract cube region from alternative volume
                alt_cube = alt_volume[z_s:z_e, y_s:y_e, x_s:x_e]
                
                # Find overlap with original cube mask
                alt_mask = alt_cube & original_cube
                
                if not np.any(alt_mask):
                    print(f"        No overlap with original mask")
                    continue
                
                # Label components in alternative within this region
                alt_labeled = label(alt_mask)
                num_alt_comps = alt_labeled.max()
                
                if num_alt_comps == 0:
                    print(f"        No components found")
                    continue
                
                print(f"        Found {num_alt_comps} component(s)")
                
                # Filter out small components
                valid_components = []
                for comp_id in range(1, num_alt_comps + 1):
                    comp_mask = (alt_labeled == comp_id)
                    comp_size = np.sum(comp_mask)
                    
                    if comp_size < min_component_size:
                        print(f"          Component {comp_id}: Too small ({comp_size} vx)")
                        continue
                    
                    valid_components.append((comp_id, comp_mask, comp_size))
                
                if len(valid_components) == 0:
                    print(f"        No valid components (all <{min_component_size} voxels)")
                    continue
                
                print(f"        Processing {len(valid_components)} valid component(s)")
                
                # Try interpolating each valid component
                interpolated_parts = []
                for comp_id, comp_mask, comp_size in valid_components:
                    print(f"          Component {comp_id}: {comp_size} voxels", end="")
                    
                    try:
                        comp_fitted = fit_curved_sheet_to_component_optimized(
                            comp_mask,
                            grid_resolution=grid_resolution,
                            thickness=thickness,
                            smoothing=smoothing,
                            max_distance=max_distance,
                            use_numba=True,
                            adaptive_resolution=True,
                            samples_per_edge=samples_per_edge
                        )
                        
                        comp_dice = calculate_dice_score(comp_mask, comp_fitted)
                        comp_coverage = calculate_coverage_score(comp_mask, comp_fitted)
                        print(f" → Dice={comp_dice:.3f}, Cov={comp_coverage:.3f}", end="")
                        
                        if comp_dice >= alt_min_dice and comp_coverage >= alt_min_coverage:
                            interpolated_parts.append(comp_fitted)
                            print(f" ✓")
                        else:
                            print(f" ✗")
                    
                    except Exception as e:
                        print(f" ✗ ({e})")
                
                if len(interpolated_parts) == 0:
                    print(f"        No parts met quality threshold")
                    continue
                
                # Combine all interpolated parts
                combined_mask = np.zeros_like(cube_mask, dtype=bool)
                for part in interpolated_parts:
                    combined_mask |= part
                
                # Check overall quality
                dice = calculate_dice_score(original_cube, combined_mask)
                coverage = calculate_coverage_score(original_cube, combined_mask)
                print(f"        Combined: Dice={dice:.3f}, Coverage={coverage:.3f}, "
                      f"{len(interpolated_parts)} part(s)")
                
                if dice > best_metrics['dice']:
                    best_result = combined_mask
                    best_metrics = {
                        'dice': dice,
                        'coverage': coverage,
                        'attempt': alt_idx + 2,
                        'num_parts': len(interpolated_parts),
                        'alt_idx': alt_idx
                    }
                    print(f"        ✓ New best result!")
                
                # If we meet the threshold, we can stop
                if dice >= alt_min_dice and coverage >= alt_min_coverage:
                    print(f"        ✓ Alternative threshold met, stopping search")
                    break
    
    # ========================================================================
    # Use best result found
    # ========================================================================
    if best_result is None:
        return component, False, {'dice': 0.0, 'coverage': 0.0, 'reason': 'no_valid_result'}
    
    interpolated_cube = best_result
    final_dice = best_metrics['dice']
    final_coverage = best_metrics['coverage']
    
    print(f"      Final: Dice={final_dice:.3f}, Coverage={final_coverage:.3f} "
          f"(attempt {best_metrics.get('attempt', 0)}, {best_metrics.get('num_parts', 1)} part(s))")
    
    # ========================================================================
    # Create interior mask (exclude border_thickness from non-boundary faces)
    # ========================================================================
    cube_shape = cube_mask.shape
    interior_mask = np.ones(cube_shape, dtype=bool)
    
    # Z dimension borders
    if cube_shape[0] > 2 * border_thickness:
        if not at_volume_boundary['z_min']:
            interior_mask[:border_thickness, :, :] = False
        if not at_volume_boundary['z_max']:
            interior_mask[-border_thickness:, :, :] = False
    else:
        bt = max(1, cube_shape[0] // 4)
        if not at_volume_boundary['z_min']:
            interior_mask[:bt, :, :] = False
        if not at_volume_boundary['z_max']:
            interior_mask[-bt:, :, :] = False
    
    # Y dimension borders
    if cube_shape[1] > 2 * border_thickness:
        if not at_volume_boundary['y_min']:
            interior_mask[:, :border_thickness, :] = False
        if not at_volume_boundary['y_max']:
            interior_mask[:, -border_thickness:, :] = False
    else:
        bt = max(1, cube_shape[1] // 4)
        if not at_volume_boundary['y_min']:
            interior_mask[:, :bt, :] = False
        if not at_volume_boundary['y_max']:
            interior_mask[:, -bt:, :] = False
    
    # X dimension borders
    if cube_shape[2] > 2 * border_thickness:
        if not at_volume_boundary['x_min']:
            interior_mask[:, :, :border_thickness] = False
        if not at_volume_boundary['x_max']:
            interior_mask[:, :, -border_thickness:] = False
    else:
        bt = max(1, cube_shape[2] // 4)
        if not at_volume_boundary['x_min']:
            interior_mask[:, :, :bt] = False
        if not at_volume_boundary['x_max']:
            interior_mask[:, :, -bt:] = False
    
    # Replace ONLY interior voxels in the original component
    result = component.copy()
    
    # Clear interior region
    result[z_s:z_e, y_s:y_e, x_s:x_e][interior_mask] = False
    
    # Place interpolated interior
    result[z_s:z_e, y_s:y_e, x_s:x_e][interior_mask] = interpolated_cube[interior_mask]
    
    interior_voxels = np.sum(interior_mask)
    replaced_voxels = np.sum(interpolated_cube[interior_mask])
    print(f"      Replaced {replaced_voxels}/{interior_voxels} interior voxels")
    
    # Report boundary preservation
    boundaries_preserved = []
    if not at_volume_boundary['z_min']:
        boundaries_preserved.append('z_min')
    if not at_volume_boundary['z_max']:
        boundaries_preserved.append('z_max')
    if not at_volume_boundary['y_min']:
        boundaries_preserved.append('y_min')
    if not at_volume_boundary['y_max']:
        boundaries_preserved.append('y_max')
    if not at_volume_boundary['x_min']:
        boundaries_preserved.append('x_min')
    if not at_volume_boundary['x_max']:
        boundaries_preserved.append('x_max')
    
    if boundaries_preserved:
        print(f"      Preserved borders: {', '.join(boundaries_preserved)}")
    else:
        print(f"      No borders preserved (all at volume boundary)")
    
    success = (final_dice >= min_dice and final_coverage >= min_coverage) or \
              (final_dice >= alt_min_dice and final_coverage >= alt_min_coverage)
    
    metrics = {
        'dice': final_dice,
        'coverage': final_coverage,
        'attempt': best_metrics.get('attempt', 0),
        'num_parts': best_metrics.get('num_parts', 1),
        'replaced_voxels': replaced_voxels,
        'interior_voxels': interior_voxels
    }
    
    return result, success, metrics


def process_component_patchwise(
    component,
    component_id,
    alternative_volumes=None,
    cube_size=32,
    overlap=8,
    border_thickness=4,
    grid_resolution=80,
    thickness=3,
    smoothing=2.0,
    max_distance=10,
    samples_per_edge=8,
    min_cube_voxels=20,
    min_dice=0.65,
    min_coverage=0.65,
    alt_min_dice=0.60,
    alt_min_coverage=0.60,
    min_component_size=100
):
    """
    Main function: Process a component using patch-wise topology checking.
    
    Workflow:
    1. Divide component into cubes
    2. Check each cube for holes (Euler number)
    3. Group overlapping/adjacent cubes with holes
    4. Interpolate grouped regions (or single cubes) with quality checking
    5. If quality low, try alternative volumes (pre-segmented at different thresholds)
    6. Filter out small components (<100 voxels) from alternatives
    7. Only replace interior, preserve borders for connectivity (except at volume edges)
    
    Parameters:
    -----------
    component : ndarray (bool)
        Binary mask of component to process
    component_id : int
        ID for logging
    alternative_volumes : list of ndarray (bool), optional
        Pre-computed segmentations at different thresholds
    cube_size : int
        Size of cube patches
    overlap : int
        Overlap between adjacent cubes
    border_thickness : int
        Border region to preserve (voxels from each face, except at volume boundary)
    min_dice : float
        Minimum Dice score required for main interpolation
    min_coverage : float
        Minimum coverage score required for main interpolation
    alt_min_dice : float
        Minimum Dice score for alternative volumes
    alt_min_coverage : float
        Minimum coverage score for alternative volumes
    min_component_size : int
        Minimum voxels for components to keep (default: 100)
    
    Returns:
    --------
    tuple (result, success_rate, metrics)
        result: Modified component with holes interpolated
        success_rate: Fraction of problematic cubes successfully interpolated
        metrics: Summary statistics
    """
    print(f"\n{'='*70}")
    print(f"Processing component {component_id} (patch-wise)")
    print(f"{'='*70}")
    
    # Step 1: Divide into cubes
    print(f"Step 1: Dividing into cubes (size={cube_size}, overlap={overlap})...")
    cubes = divide_component_into_cubes(component, cube_size=cube_size, overlap=overlap)
    print(f"  Generated {len(cubes)} cubes")
    
    if len(cubes) == 0:
        print("  No cubes generated (component too small?)")
        return component, 0.0, {'total_cubes': 0}
    
    # Step 2: Check topology of each cube
    print(f"\nStep 2: Checking topology of each cube...")
    cubes_with_holes = []
    cubes_with_holes_indices = []
    
    for i, cube_info in enumerate(cubes):
        has_hole, beta1, chi = check_cube_topology(cube_info['mask'], min_voxels=min_cube_voxels)
        
        if has_hole:
            print(f"  Cube {i} at {cube_info['center']}: β1={beta1} (χ={chi}) ⚠ HAS HOLE")
            cubes_with_holes.append(cube_info)
            cubes_with_holes_indices.append(i)
        else:
            print(f"  Cube {i} at {cube_info['center']}: β1={beta1} (χ={chi}) ✓ OK")
    
    if len(cubes_with_holes) == 0:
        print("\n✓ No cubes have holes - component is topologically correct!")
        return component, 1.0, {
            'total_cubes': len(cubes),
            'problematic_cubes': 0,
            'successful_interpolations': 0
        }
    
    print(f"\nFound {len(cubes_with_holes)} cube(s) with holes")
    
    # Step 3: Group overlapping/adjacent cubes
    print(f"\nStep 3: Grouping overlapping/adjacent cubes...")
    groups = find_overlapping_cube_groups(cubes_with_holes)
    print(f"  Found {len(groups)} group(s)")
    
    for g_idx, group in enumerate(groups):
        print(f"  Group {g_idx + 1}: {len(group)} cube(s)")
    
    # Step 4: Interpolate each group with quality checking
    print(f"\nStep 4: Interpolating cube groups with quality checking...")
    result = component.copy()
    successful_interpolations = 0
    failed_interpolations = 0
    all_metrics = []
    
    for g_idx, group_indices in enumerate(groups):
        print(f"\n  Group {g_idx + 1}/{len(groups)}: {len(group_indices)} cube(s)")
        
        if len(group_indices) == 1:
            # Single cube - interpolate just that cube
            cube_idx = group_indices[0]
            cube_coords = cubes_with_holes[cube_idx]['coords']
            at_boundary = cubes_with_holes[cube_idx]['at_volume_boundary']
            
            print(f"    Single cube - interpolating independently")
            result, success, metrics = interpolate_cube_region_with_quality_check(
                result,
                cube_coords,
                at_boundary,
                alternative_volumes=alternative_volumes,
                border_thickness=border_thickness,
                grid_resolution=grid_resolution,
                thickness=thickness,
                smoothing=smoothing,
                max_distance=max_distance,
                samples_per_edge=samples_per_edge,
                min_dice=min_dice,
                min_coverage=min_coverage,
                alt_min_dice=alt_min_dice,
                alt_min_coverage=alt_min_coverage,
                min_component_size=min_component_size
            )
            
            all_metrics.append(metrics)
            if success:
                successful_interpolations += 1
                print(f"    ✓ Interpolation successful")
            else:
                failed_interpolations += 1
                print(f"    ✗ Interpolation failed or low quality")
        else:
            # Multiple cubes - merge into single region
            merged_coords, merged_boundary = merge_cube_regions(
                cubes_with_holes,
                group_indices,
                component.shape
            )
            
            print(f"    Multiple cubes - merging into region: "
                  f"z[{merged_coords[0]}:{merged_coords[1]}], "
                  f"y[{merged_coords[2]}:{merged_coords[3]}], "
                  f"x[{merged_coords[4]}:{merged_coords[5]}]")
            
            result, success, metrics = interpolate_cube_region_with_quality_check(
                result,
                merged_coords,
                merged_boundary,
                alternative_volumes=alternative_volumes,
                border_thickness=border_thickness,
                grid_resolution=grid_resolution,
                thickness=thickness,
                smoothing=smoothing,
                max_distance=max_distance,
                samples_per_edge=samples_per_edge,
                min_dice=min_dice,
                min_coverage=min_coverage,
                alt_min_dice=alt_min_dice,
                alt_min_coverage=alt_min_coverage,
                min_component_size=min_component_size
            )
            
            all_metrics.append(metrics)
            if success:
                successful_interpolations += 1
                print(f"    ✓ Interpolation successful")
            else:
                failed_interpolations += 1
                print(f"    ✗ Interpolation failed or low quality")
    
    success_rate = successful_interpolations / len(groups) if len(groups) > 0 else 0.0
    
    summary_metrics = {
        'total_cubes': len(cubes),
        'problematic_cubes': len(cubes_with_holes),
        'groups': len(groups),
        'successful_interpolations': successful_interpolations,
        'failed_interpolations': failed_interpolations,
        'success_rate': success_rate,
        'avg_dice': np.mean([m['dice'] for m in all_metrics]) if all_metrics else 0.0,
        'avg_coverage': np.mean([m['coverage'] for m in all_metrics]) if all_metrics else 0.0
    }
    
    print(f"\n✓ Component {component_id} processing complete")
    print(f"  Success rate: {success_rate:.1%} ({successful_interpolations}/{len(groups)} groups)")
    print(f"  Avg Dice: {summary_metrics['avg_dice']:.3f}")
    print(f"  Avg Coverage: {summary_metrics['avg_coverage']:.3f}")
    
    return result, success_rate, summary_metrics


# ============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# ============================================================================

def process_multiple_components_patchwise(
    volume,
    alternative_volumes=None,
    cube_size=32,
    overlap=8,
    border_thickness=4,
    grid_resolution=80,
    thickness=3,
    smoothing=2.0,
    max_distance=10,
    samples_per_edge=8,
    min_dice=0.65,
    min_coverage=0.65,
    alt_min_dice=0.60,
    alt_min_coverage=0.60,
    min_component_size=100,
    use_parallel=True,
    n_jobs=-1
):
    """
    Process all components in a volume using patch-wise approach with quality checking.
    
    Parameters:
    -----------
    volume : ndarray (bool)
        Binary volume to process
    alternative_volumes : list of ndarray (bool), optional
        Pre-computed segmentations at different thresholds (e.g., different sigma values)
        These are used when main interpolation quality is too low
    cube_size : int
        Size of cube patches
    overlap : int
        Overlap between adjacent cubes
    min_dice : float
        Minimum Dice score for main interpolation
    min_coverage : float
        Minimum coverage score for main interpolation
    alt_min_dice : float
        Minimum Dice score for alternative volumes
    alt_min_coverage : float
        Minimum coverage score for alternative volumes
    min_component_size : int
        Minimum voxels for components in alternative volumes (filters out noise)
    """
    print(f"\n{'='*70}")
    print(f"PATCH-WISE COMPONENT PROCESSING WITH QUALITY CHECKING")
    print(f"{'='*70}")
    
    labeled_volume = label(volume)
    num_components = labeled_volume.max()
    
    print(f"Found {num_components} components")
    print(f"Cube parameters: size={cube_size}, overlap={overlap}, border={border_thickness}")
    print(f"Main thresholds: Dice≥{min_dice:.2f}, Coverage≥{min_coverage:.2f}")
    print(f"Alternative thresholds: Dice≥{alt_min_dice:.2f}, Coverage≥{alt_min_coverage:.2f}")
    print(f"Min component size: {min_component_size} voxels")
    if alternative_volumes is not None:
        print(f"Using {len(alternative_volumes)} alternative volume(s)")
    
    result_labeled = np.zeros_like(volume, dtype=np.int32)
    all_success_rates = []
    all_metrics = []
    
    for i in range(1, num_components + 1):
        component_mask = (labeled_volume == i)
        
        # Process component patch-wise
        processed_component, success_rate, metrics = process_component_patchwise(
            component_mask,
            component_id=i,
            alternative_volumes=alternative_volumes,
            cube_size=cube_size,
            overlap=overlap,
            border_thickness=border_thickness,
            grid_resolution=grid_resolution,
            thickness=thickness,
            smoothing=smoothing,
            max_distance=max_distance,
            samples_per_edge=samples_per_edge,
            min_dice=min_dice,
            min_coverage=min_coverage,
            alt_min_dice=alt_min_dice,
            alt_min_coverage=alt_min_coverage,
            min_component_size=min_component_size
        )
        
        # Assign to result
        result_labeled[processed_component] = i
        all_success_rates.append(success_rate)
        all_metrics.append(metrics)
    
    result_binary = result_labeled > 0
    
    # Overall statistics
    avg_success_rate = np.mean(all_success_rates) if all_success_rates else 0.0
    total_problematic_cubes = sum(m.get('problematic_cubes', 0) for m in all_metrics)
    total_successful = sum(m.get('successful_interpolations', 0) for m in all_metrics)
    
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Components processed: {num_components}")
    print(f"Final components: {label(result_binary).max()}")
    print(f"Total problematic cubes: {total_problematic_cubes}")
    print(f"Successfully interpolated: {total_successful}")
    print(f"Average success rate: {avg_success_rate:.1%}")
    
    return result_binary, result_labeled, all_metrics





# Adapted from: Kaggle notebook, Cell 4
# https://www.kaggle.com/code/mariusheuser/local-interpolation-interference
#
# Modifications:
# - The original function used a global configuration.
# - This version accepts a custom configuration. Values missing from the
#   custom configuration are filled in from the original global configuration.
# - The original GPU configuration was replaced with automatic device
#   selection based on the available hardware:
#   - Two or more GPUs: Model 1 and Model 2 are assigned to separate GPUs
#     and run in parallel.
#   - One GPU: Both models use the same GPU.
#   - No GPU: Both models run on the CPU.


# ============================================================
# OPTIMIZED DUAL-GPU ENSEMBLE INFERENCE
# Model 1 on GPU:0, Model 2 on GPU:1, Parallel Processing
# ============================================================

# ============================================================
# 1. INFERENCE CONFIGURATION
# ============================================================

CONFIG = {
    # --- MODEL 1 (GPU:0) ---
    "model1_folder": "/kaggle/input/models/nguyncdngs/nnunet-resenc/pytorch/default/3/nnUNet_ResEnc",
    "model1_checkpoint": "checkpoint_final.pth",
    "model1_weight": 0.6,
    "model1_device": "cuda:0",  # First T4
    
    # --- MODEL 2 (GPU:1) ---
    "model2_folder": "/kaggle/input/models/nguyncdngs/nnunet-resenc/pytorch/default/2/nnUNet_ResEnc",
    "model2_checkpoint": "checkpoint_final.pth",
    "model2_weight": 0.4,
    "model2_device": "cuda:1",  # Second T4
    
    # Đường dẫn data
    "input_dir": "/kaggle/input/vesuvius-challenge-surface-detection/test_images",
    "output_zip": "submission.zip",
    
    # Cấu hình Predictor
    "use_folds": ("all",),          
    "tile_step_size": 0.5,      
    "use_gaussian": True,        
    "use_mirroring": True,
    
    # Spacing
    "spacing": [1.0, 1.0, 1.0],

    # --- POST-PROCESSING CONFIG ---
    "enable_postproc": True,
    "T_low": 0.2,
    "T_high": 0.83,
    "z_radius": 1,
    "xy_radius": 0,
    "min_object_size": 2000,
}

# ============================================================
# 2. IMPORTS
# ============================================================

import os
import glob
import zipfile
import numpy as np
import torch
import tifffile
from tqdm import tqdm
from skimage import morphology
import scipy.ndimage as ndi
from concurrent.futures import ThreadPoolExecutor
import threading

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.tif_reader_writer import Tiff3DIO

# ============================================================
# 3. POST-PROCESSING FUNCTIONS
# ============================================================

def build_anisotropic_struct(z_radius: int, xy_radius: int):
    """Tạo kernel 3D hình elip/trụ để đóng lỗ hổng."""
    z, r = z_radius, xy_radius

    if z == 0 and r == 0:
        return None

    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct

    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct

    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct

def topo_postprocess(
                    probs,          # (D, H, W)
                    T_low=0.6,
                    T_high=0.9,
                    z_radius=1,
                    xy_radius=1,
                    dust_min_size=500,
                ):

    # --- Step 1: 3D Hysteresis ---
    strong = probs >= T_high
    weak   = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # --- Step 2: 3D Anisotropic Closing ---
    struct_close = build_anisotropic_struct(z_radius, xy_radius)
    if struct_close is not None:
        mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: Dust Removal
    if dust_min_size > 0:
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)

# ============================================================
# 4. PARALLEL PREDICTION CLASS
# ============================================================

class ParallelPredictor:
    """Wrapper to run predictions in parallel on separate GPUs."""
    
    def __init__(self, predictor, device_name):
        self.predictor = predictor
        self.device_name = device_name
        self.lock = threading.Lock()
    
    def predict(self, image, properties):
        """Thread-safe prediction on assigned GPU."""
        with self.lock:
            ret = self.predictor.predict_single_npy_array(
                image, 
                properties, 
                segmentation_previous_stage=None, 
                output_file_truncated=None, 
                save_or_return_probabilities=True
            )
        return ret

# ============================================================
# 5. MAIN INFERENCE ENGINE
# ============================================================

def run_inference(config):
    print("--> [INIT] Setting up Dual-GPU Parallel Ensemble Inference...")

    config = CONFIG | config
    
    # # 1. Check GPU availability
    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA not available!")
    
    # gpu_count = torch.cuda.device_count()
    # print(f"--> [GPU CHECK] Found {gpu_count} GPU(s)")
    
    # if gpu_count < 2:
    #     print(f"--> [WARNING] Only {gpu_count} GPU found. Falling back to single GPU mode.")
    #     device1 = torch.device('cuda:0')
    #     device2 = torch.device('cuda:0')
    # else:
    #     device1 = torch.device(config["model1_device"])
    #     device2 = torch.device(config["model2_device"])
    #     print(f"--> [GPU ASSIGNMENT] Model 1 -> {device1}, Model 2 -> {device2}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"--> [GPU CHECK] Found {gpu_count} GPU(s)")

        if gpu_count >= 2:
            # Zwei GPUs -> echtes paralleles Ensemble
            device1 = torch.device("cuda:0")
            device2 = torch.device("cuda:1")
            parallel = True

        else:
            # Eine GPU -> beide Modelle auf derselben GPU
            device1 = torch.device("cuda:0")
            device2 = torch.device("cuda:0")
            parallel = False

    else:
        # Keine GPU -> CPU
        print("--> [CPU MODE] CUDA not available.")
        device1 = torch.device("cpu")
        device2 = torch.device("cpu")
        parallel = False

    print(f"--> [DEVICE] Model 1 -> {device1}")
    print(f"--> [DEVICE] Model 2 -> {device2}")
    print(f"--> [MODE] Parallel = {parallel}")

    # 2. Initialize Model 1 on GPU:0
    print(f"--> [MODEL 1] Loading on {device1}...")
    predictor1 = nnUNetPredictor(
        tile_step_size=config["tile_step_size"],
        use_gaussian=config["use_gaussian"],
        use_mirroring=config["use_mirroring"],
        perform_everything_on_device=True,
        device=device1,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    
    predictor1.initialize_from_trained_model_folder(
        config["model1_folder"],
        use_folds=config["use_folds"],
        checkpoint_name=config["model1_checkpoint"]
    )
    print(f"   -> Model 1 loaded on {device1}")

    # 3. Initialize Model 2 on GPU:1
    print(f"--> [MODEL 2] Loading on {device2}...")
    predictor2 = nnUNetPredictor(
        tile_step_size=config["tile_step_size"],
        use_gaussian=config["use_gaussian"],
        use_mirroring=config["use_mirroring"],
        perform_everything_on_device=True,
        device=device2,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    
    predictor2.initialize_from_trained_model_folder(
        config["model2_folder"],
        use_folds=config["use_folds"],
        checkpoint_name=config["model2_checkpoint"]
    )
    print(f"   -> Model 2 loaded on {device2}")
    
    # 4. Wrap predictors for parallel execution
    parallel_pred1 = ParallelPredictor(predictor1, device1)
    parallel_pred2 = ParallelPredictor(predictor2, device2)
    
    # 5. Setup Image Reader
    reader = Tiff3DIO()
    
    # 6. Prepare Input Files
    test_files = sorted(glob.glob(os.path.join(config["input_dir"], "*.tif")))
    if not test_files:
        print("--> [WARNING] No .tif files found.")
        return 
    
    print(f"--> [DATA] Found {len(test_files)} files to process.")
    print(f"--> [ENSEMBLE] Weights: Model1={config['model1_weight']}, Model2={config['model2_weight']}")

    # 7. Processing Loop with Parallel Execution
    with zipfile.ZipFile(config["output_zip"], 'w', zipfile.ZIP_DEFLATED) as zf:
        
        for file_path in tqdm(test_files, desc="Parallel Dual-GPU Inference"):
            filename = os.path.basename(file_path)
            
            # Read image
            image, _ = reader.read_images([file_path])
            
            properties = {
                'spacing': config['spacing']
            }

            # --- PARALLEL PREDICTION ON BOTH GPUS ---
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both predictions simultaneously
                future1 = executor.submit(parallel_pred1.predict, image, properties)
                future2 = executor.submit(parallel_pred2.predict, image, properties)
                
                # Wait for both to complete
                ret1 = future1.result()
                ret2 = future2.result()

            # --- Extract Probabilities ---
            if isinstance(ret1, tuple):
                _, probabilities1 = ret1
            else:
                probabilities1 = None

            if isinstance(ret2, tuple):
                _, probabilities2 = ret2
            else:
                probabilities2 = None

            # --- ENSEMBLE PROBABILITIES ---
            if probabilities1 is not None and probabilities2 is not None:
                # Weighted ensemble
                total_weight = config["model1_weight"] + config["model2_weight"]
                w1 = config["model1_weight"] / total_weight
                w2 = config["model2_weight"] / total_weight
                
                ensembled_probs = w1 * probabilities1 + w2 * probabilities2
                prob_map = ensembled_probs[1]
                # Apply multiple threshold levels
                pred = topo_postprocess(prob_map, T_low=0.2,
                                            T_high=0.83,
                                            z_radius=1,
                                            xy_radius=0,
                                            dust_min_size=100)
                pred = zero_volume_faces(pred.astype(bool), thickness=3)
                pred = morphology.remove_small_objects(
                    pred.astype(bool),
                    min_size=1000,
                    connectivity=3
                )

                pred2 = topo_postprocess(prob_map, T_low=0.5,
                                            T_high=0.83,
                                            z_radius=1,
                                            xy_radius=0,
                                            dust_min_size=100)
                pred2 = zero_volume_faces(pred2.astype(bool), thickness=3)
                pred2 = morphology.remove_small_objects(
                    pred2.astype(bool),
                    min_size=1000,
                    connectivity=3
                )
                
                pred3 = topo_postprocess(prob_map, T_low=0.6,
                                            T_high=0.83,
                                            z_radius=1,
                                            xy_radius=0,
                                            dust_min_size=100)
                pred3 = zero_volume_faces(pred3.astype(bool), thickness=3)
                pred3 = morphology.remove_small_objects(
                    pred3.astype(bool),
                    min_size=1000,
                    connectivity=3
                )
                
                pred5 = topo_postprocess(prob_map, T_low=0.7,
                                            T_high=0.83,
                                            z_radius=1,
                                            xy_radius=0,
                                            dust_min_size=100)
                pred5 = zero_volume_faces(pred5.astype(bool), thickness=3)
                pred5 = morphology.remove_small_objects(
                    pred5.astype(bool),
                    min_size=1000,
                    connectivity=3
                )
                pred1_uint8 = pred2.astype(np.uint8)
                pred2_uint8 = pred2.astype(np.uint8)
                pred3_uint8 = pred3.astype(np.uint8)
                pred5_uint8 = pred5.astype(np.uint8)
                # Fill holes
                pred = binary_fill_holes(pred.astype(bool))
                
                # Crop borders
                pred = pred[3:-3, 3:-3, 3:-3]
                pred2 = pred2[3:-3, 3:-3, 3:-3]
                pred3 = pred3[3:-3, 3:-3, 3:-3]
                pred5 = pred5[3:-3, 3:-3, 3:-3]
                
                # Advanced component processing
                pred, masks, metrics = process_multiple_components_patchwise(
                    volume=pred,
                    alternative_volumes=[pred2, pred3, pred5],  # Pass pre-computed alternatives
                    cube_size=80,
                    overlap=40,
                    border_thickness=8,
                    grid_resolution=80,
                    thickness=3,
                    smoothing=1.0,
                    samples_per_edge=8,
                    min_dice=0.75,           # Main thresholds
                    min_coverage=0.75,
                    alt_min_dice=0.45,       # More lenient for alternatives
                    alt_min_coverage=0.80,
                    min_component_size=100,  # Filter small noise
                    use_parallel=True,
                    n_jobs=-1
                )
                
                # Final cleanup
                pred = morphology.remove_small_objects(
                    pred, 
                    min_size=1000, 
                    connectivity=3
                )
                
                # Restore border padding
                border = 3
                pred = np.pad(
                    pred,
                    pad_width=(
                        (border, border),
                        (border, border),
                        (border, border)
                    ),
                    mode="constant",
                    constant_values=0
                )
                
                pred_uint8 = pred.astype(np.uint8)
                
            else:
                # Fallback without post-processing
                if segmentation_mask.ndim == 4:
                    segmentation_mask = segmentation_mask[0]
                pred_uint8 = segmentation_mask.astype(np.uint8)

            # --- Save to Zip ---
            temp_output_path = filename 
            tifffile.imwrite("base.tif", pred1_uint8)
            tifffile.imwrite("pred2.tif", pred2_uint8)
            tifffile.imwrite("pred3.tif", pred3_uint8)
            tifffile.imwrite("pred5.tif", pred5_uint8)
            tifffile.imwrite(temp_output_path, pred_uint8)
            zf.write(temp_output_path, arcname=filename)
            
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

    print(f"\n--> [DONE] Parallel Dual-GPU Inference Complete!")
    print(f"    Output: {config['output_zip']}")

if __name__ == "__main__":
    run_inference()