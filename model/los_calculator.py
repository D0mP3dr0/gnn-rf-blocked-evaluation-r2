

"""Line-of-sight (LOS) calculation for RF propagation."""

import numpy as np
import torch
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import math


@dataclass
class LOSResult:
    """Result of a line-of-sight analysis."""
    is_los: bool
    obstruction_height_m: float
    fresnel_clearance_pct: float
    obstruction_distance_m: float
    num_obstructions: int
    path_length_m: float


@dataclass
class PathProfile:
    """Perfil de caminho entre TX e RX."""
    distances_m: np.ndarray
    elevations_m: np.ndarray
    canopy_m: np.ndarray
    surface_m: np.ndarray
    los_heights_m: np.ndarray
    clearances_m: np.ndarray
    fresnel_radii_m: np.ndarray


SPEED_OF_LIGHT = 299792458.0  # m/s
EARTH_RADIUS = 6371000.0


def calculate_wavelength(frequency_mhz: float) -> float:
    """Calcula comprimento de onda em metros."""
    return SPEED_OF_LIGHT / (frequency_mhz * 1e6)


def fresnel_zone_radius(
    d1_m: Union[float, np.ndarray],
    d2_m: Union[float, np.ndarray],
    frequency_mhz: float,
    zone: int = 1
) -> Union[float, np.ndarray]:
    """
    Computes the Fresnel zone radius.

    r_n = sqrt(n * lambda * d1 * d2 / (d1 + d2))

    Args:
        d1_m: distance TX -> point (m).
        d2_m: distance point -> RX (m).
        frequency_mhz: frequency in MHz.
        zone: Fresnel zone number (default: 1).

    Returns:
        Fresnel zone radius in meters.
    """
    wavelength_m = calculate_wavelength(frequency_mhz)
    d_total = d1_m + d2_m


    d_total = np.maximum(d_total, 1.0)

    radius = np.sqrt(zone * wavelength_m * d1_m * d2_m / d_total)
    return radius


def earth_curvature_correction(distance_m: Union[float, np.ndarray], k: float = 4/3) -> Union[float, np.ndarray]:
    """
    Computes the Earth curvature correction.

    h_correction = d^2 / (2 * k * R)

    Args:
        distance_m: distance from the reference point (m).
        k: effective curvature factor (4/3 for a standard atmosphere).

    Returns:
        Height correction in meters (added to elevation).
    """
    return (distance_m ** 2) / (2 * k * EARTH_RADIUS)


def interpolate_path_profile(
    tx_pos: Tuple[float, float, float],
    rx_pos: Tuple[float, float, float],
    dem_grid: np.ndarray,
    canopy_grid: np.ndarray,
    transform: 'rasterio.Affine',
    num_points: int = 100,
    frequency_mhz: float = 1800.0,
    apply_earth_curvature: bool = True
) -> PathProfile:
    """
    Interpolates the elevation profile along the TX-RX path.

    Args:
        tx_pos: (lon, lat, height_m) of the transmitter.
        rx_pos: (lon, lat, height_m) of the receiver.
        dem_grid: elevation (DEM) grid [H, W].
        canopy_grid: canopy height grid [H, W].
        transform: raster affine transform.
        num_points: number of sample points.
        frequency_mhz: frequency used for the Fresnel calculation.
        apply_earth_curvature: whether to apply the Earth curvature correction.

    Returns:
        PathProfile with all path data.
    """
    tx_lon, tx_lat, tx_height = tx_pos
    rx_lon, rx_lat, rx_height = rx_pos


    t = np.linspace(0, 1, num_points)
    lons = tx_lon + t * (rx_lon - tx_lon)
    lats = tx_lat + t * (rx_lat - tx_lat)


    dx = (rx_lon - tx_lon) * 111320 * np.cos(np.radians((tx_lat + rx_lat) / 2))
    dy = (rx_lat - tx_lat) * 110540
    total_distance = np.sqrt(dx**2 + dy**2)
    distances = t * total_distance


    elevations = np.zeros(num_points)
    canopy = np.zeros(num_points)

    for i, (lon, lat) in enumerate(zip(lons, lats)):

        col = int((lon - transform.c) / transform.a)
        row = int((lat - transform.f) / transform.e)


        if 0 <= row < dem_grid.shape[0] and 0 <= col < dem_grid.shape[1]:
            elevations[i] = dem_grid[row, col]
            canopy[i] = canopy_grid[row, col] if canopy_grid is not None else 0
        else:
            elevations[i] = 0
            canopy[i] = 0


    surface = elevations + canopy


    if apply_earth_curvature:

        mid_distance = total_distance / 2
        curvature = earth_curvature_correction(np.abs(distances - mid_distance))
        surface = surface + curvature


    los_heights = tx_height + t * (rx_height - tx_height)


    tx_total = elevations[0] + tx_height
    rx_total = elevations[-1] + rx_height
    los_heights = tx_total + t * (rx_total - tx_total)


    clearances = los_heights - surface


    d1 = distances
    d2 = total_distance - distances
    fresnel_radii = fresnel_zone_radius(d1, d2, frequency_mhz)

    return PathProfile(
        distances_m=distances,
        elevations_m=elevations,
        canopy_m=canopy,
        surface_m=surface,
        los_heights_m=los_heights,
        clearances_m=clearances,
        fresnel_radii_m=fresnel_radii
    )


def check_line_of_sight(
    tx_pos: Tuple[float, float, float],
    rx_pos: Tuple[float, float, float],
    dem_grid: np.ndarray,
    canopy_grid: Optional[np.ndarray] = None,
    transform: Optional['rasterio.Affine'] = None,
    frequency_mhz: float = 1800.0,
    num_points: int = 100,
    fresnel_clearance_required: float = 0.6
) -> LOSResult:
    """
    Checks whether there is line of sight between TX and RX.

    Args:
        tx_pos: (lon, lat, height_m) of the transmitter.
        rx_pos: (lon, lat, height_m) of the receiver.
        dem_grid: elevation grid [H, W].
        canopy_grid: canopy height grid [H, W] (optional).
        transform: raster affine transform.
        frequency_mhz: frequency used for the Fresnel calculation.
        num_points: number of sample points along the path.
        fresnel_clearance_required: minimum required clearance fraction (0.6 = 60%).

    Returns:
        LOSResult with detailed information.
    """
    if canopy_grid is None:
        canopy_grid = np.zeros_like(dem_grid)

    if transform is None:

        from rasterio.transform import Affine
        transform = Affine(0.0003, 0, tx_pos[0] - 0.01, 0, -0.0003, tx_pos[1] + 0.01)


    profile = interpolate_path_profile(
        tx_pos, rx_pos, dem_grid, canopy_grid,
        transform, num_points, frequency_mhz
    )


    required_clearance = fresnel_clearance_required * profile.fresnel_radii_m
    effective_clearance = profile.clearances_m - required_clearance


    check_range = slice(1, -1)

    obstructions = effective_clearance[check_range] < 0
    num_obstructions = np.sum(obstructions)

    if num_obstructions > 0:

        obstruction_indices = np.where(obstructions)[0] + 1
        obstruction_heights = -effective_clearance[check_range][obstructions]
        max_obstruction_idx = obstruction_indices[np.argmax(obstruction_heights)]

        max_obstruction_height = np.max(obstruction_heights)
        obstruction_distance = profile.distances_m[max_obstruction_idx]


        min_clearance_pct = np.min(profile.clearances_m[check_range] /
                                   np.maximum(profile.fresnel_radii_m[check_range], 0.1))

        return LOSResult(
            is_los=False,
            obstruction_height_m=max_obstruction_height,
            fresnel_clearance_pct=min_clearance_pct * 100,
            obstruction_distance_m=obstruction_distance,
            num_obstructions=int(num_obstructions),
            path_length_m=profile.distances_m[-1]
        )
    else:

        min_clearance_pct = np.min(profile.clearances_m[check_range] /
                                   np.maximum(profile.fresnel_radii_m[check_range], 0.1))

        return LOSResult(
            is_los=True,
            obstruction_height_m=0.0,
            fresnel_clearance_pct=min_clearance_pct * 100,
            obstruction_distance_m=0.0,
            num_obstructions=0,
            path_length_m=profile.distances_m[-1]
        )


def check_los_batch(
    tx_pos: Tuple[float, float, float],
    rx_positions: torch.Tensor,
    dem_tensor: torch.Tensor,
    canopy_tensor: Optional[torch.Tensor] = None,
    cell_size_m: float = 30.0,
    frequency_mhz: float = 1800.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Checks LOS for multiple receivers in a batch (optimized for GNN use).

    Args:
        tx_pos: (x, y, height_m) of the transmitter.
        rx_positions: [N, 2] or [N, 3] receiver positions.
        dem_tensor: [H, W] terrain elevation.
        canopy_tensor: [H, W] canopy height (optional).
        cell_size_m: cell size in meters.
        frequency_mhz: frequency in MHz.

    Returns:
        is_los: [N] bool tensor.
        obstruction_height: [N] obstruction height in meters.
    """
    device = rx_positions.device
    n_rx = rx_positions.shape[0]

    tx_x, tx_y, tx_h = tx_pos

    if canopy_tensor is None:
        canopy_tensor = torch.zeros_like(dem_tensor)


    surface = dem_tensor + canopy_tensor

    is_los = torch.ones(n_rx, dtype=torch.bool, device=device)
    obstruction_heights = torch.zeros(n_rx, device=device)


    for i in range(n_rx):
        rx_x = rx_positions[i, 0].item()
        rx_y = rx_positions[i, 1].item()
        rx_h = rx_positions[i, 2].item() if rx_positions.shape[1] > 2 else 1.5


        dx = rx_x - tx_x
        dy = rx_y - tx_y
        distance = math.sqrt(dx**2 + dy**2) * cell_size_m

        if distance < cell_size_m:
            continue


        num_samples = int(distance / cell_size_m) + 1
        t = torch.linspace(0, 1, num_samples, device=device)

        sample_x = tx_x + t * dx
        sample_y = tx_y + t * dy


        col_idx = sample_x.long()
        row_idx = sample_y.long()


        valid = (row_idx >= 0) & (row_idx < surface.shape[0]) & \
                (col_idx >= 0) & (col_idx < surface.shape[1])

        if not valid.any():
            continue


        row_idx = torch.clamp(row_idx, 0, surface.shape[0] - 1)
        col_idx = torch.clamp(col_idx, 0, surface.shape[1] - 1)
        surface_heights = surface[row_idx, col_idx]


        tx_total_h = surface[int(tx_y), int(tx_x)].item() + tx_h
        rx_total_h = surface[int(rx_y), int(rx_x)].item() + rx_h
        los_heights = tx_total_h + t * (rx_total_h - tx_total_h)


        if num_samples > 2:
            clearances = los_heights[1:-1] - surface_heights[1:-1]
            min_clearance = clearances.min()

            if min_clearance < 0:
                is_los[i] = False
                obstruction_heights[i] = -min_clearance

    return is_los, obstruction_heights


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTE: LOS Calculator")
    print("=" * 70)


    np.random.seed(42)
    H, W = 200, 200


    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    dem = 100 + 50 * np.exp(-(X**2 + Y**2) / 0.2)


    canopy = np.random.uniform(0, 15, (H, W))
    canopy[dem > 130] = 0

    print("\n## 1. Teste de Interpolação de Path")
    print("-" * 50)

    from rasterio.transform import Affine
    transform = Affine(0.0003, 0, -47.0, 0, -0.0003, -22.0)

    tx_pos = (-47.0, -22.0, 30)
    rx_pos = (-46.95, -22.0, 1.5)

    profile = interpolate_path_profile(
        tx_pos, rx_pos, dem, canopy, transform,
        num_points=50, frequency_mhz=1800
    )

    print(f"   Path length: {profile.distances_m[-1]:.1f} m")
    print(f"   Elevation range: {profile.elevations_m.min():.1f} - {profile.elevations_m.max():.1f} m")
    print(f"   Min clearance: {profile.clearances_m.min():.1f} m")
    print(f"   Max Fresnel r1: {profile.fresnel_radii_m.max():.1f} m")

    print("\n## 2. Teste de LOS Check")
    print("-" * 50)


    result1 = check_line_of_sight(
        (-47.0, -22.0, 50),
        (-46.95, -22.0, 1.5),
        dem, canopy, transform
    )
    print(f"   Caso 1 (Torre 50m): LOS={result1.is_los}, Fresnel={result1.fresnel_clearance_pct:.1f}%")


    result2 = check_line_of_sight(
        (-47.02, -22.0, 5),
        (-46.98, -22.0, 1.5),
        dem, canopy, transform
    )
    print(f"   Caso 2 (Torre 5m): LOS={result2.is_los}, Obstrução={result2.obstruction_height_m:.1f}m")

    print("\n## 3. Teste de Fresnel Zone")
    print("-" * 50)
    for d in [1000, 5000, 10000]:
        r = fresnel_zone_radius(d/2, d/2, 1800.0)
        print(f"   {d}m total, midpoint: r1 = {r:.2f} m")

    print("\n## 4. Teste de Curvatura da Terra")
    print("-" * 50)
    for d in [5000, 10000, 20000]:
        c = earth_curvature_correction(d)
        print(f"   {d/1000:.0f}km: correção = {c:.2f} m")

    print("\n## 5. Teste Batch (Tensor)")
    print("-" * 50)


    dem_t = torch.tensor(dem, dtype=torch.float32)
    canopy_t = torch.tensor(canopy, dtype=torch.float32)


    rx_positions = torch.tensor([
        [150, 100, 1.5],
        [110, 100, 1.5],
        [50, 100, 1.5],
    ], dtype=torch.float32)

    is_los, obst_h = check_los_batch(
        (100, 100, 30), rx_positions, dem_t, canopy_t, cell_size_m=30
    )
    print(f"   is_los: {is_los.numpy()}")
    print(f"   obstruction: {obst_h.numpy().round(2)}")

    print("\n" + "=" * 70)
    print("✅ TESTES CONCLUÍDOS")
    print("=" * 70)
