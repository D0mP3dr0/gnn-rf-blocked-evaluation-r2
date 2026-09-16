

"""Physical RF propagation models."""

import numpy as np
import torch
from typing import Union, Tuple, Optional
from dataclasses import dataclass


SPEED_OF_LIGHT = 299792458.0  # m/s
DEFAULT_RSSI_THRESHOLD = -120.0  # dBm


def free_space_path_loss(
    distance_m: Union[float, np.ndarray, torch.Tensor],
    frequency_mhz: float
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes Free Space Path Loss (FSPL) via the Friis equation.

    FSPL(dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.44

    Args:
        distance_m: distance in meters (array/tensor accepted).
        frequency_mhz: frequency in MHz.

    Returns:
        Path loss in dB.

    Example:
        >>> fspl = free_space_path_loss(1000, 1800)  # 1 km, 1800 MHz
        >>> print(f"FSPL: {fspl:.2f} dB")  # ~97.5 dB
    """
    # Convert to km
    if isinstance(distance_m, torch.Tensor):
        distance_km = distance_m / 1000.0
        distance_km = torch.clamp(distance_km, min=0.001)
        return 20 * torch.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.44
    else:
        distance_km = np.asarray(distance_m) / 1000.0
        distance_km = np.clip(distance_km, 0.001, None)
        return 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.44


def fspl_with_gains(
    distance_m: float,
    frequency_mhz: float,
    tx_power_dbm: float,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0
) -> float:
    """
    Computes received power accounting for antenna gains.

    P_rx = P_tx + G_tx + G_rx - FSPL

    Args:
        distance_m: distance in meters.
        frequency_mhz: frequency in MHz.
        tx_power_dbm: transmit power in dBm.
        tx_gain_dbi: TX antenna gain in dBi.
        rx_gain_dbi: RX antenna gain in dBi.

    Returns:
        Received power in dBm.
    """
    fspl = free_space_path_loss(distance_m, frequency_mhz)
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl


def vegetation_loss_itu_p833(
    path_length_m: Union[float, np.ndarray, torch.Tensor],
    frequency_mhz: float,
    canopy_height_m: Union[float, np.ndarray, torch.Tensor] = 10.0
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes vegetation attenuation per ITU-R P.833-9.

    A = A_m * (1 - exp(-d*gamma/A_m))

    where:
        gamma = 0.39 * f^0.39  (specific attenuation coefficient, dB/m);
        A_m = maximum attenuation for very deep vegetation.

    Args:
        path_length_m: path length through vegetation (m).
        frequency_mhz: frequency in MHz.
        canopy_height_m: canopy height in meters (affects A_m).

    Returns:
        Vegetation attenuation in dB.

    Example:
        >>> loss = vegetation_loss_itu_p833(100, 1800, 15)  # 100 m path, 1800 MHz, 15 m canopy
        >>> print(f"Veg loss: {loss:.2f} dB")
    """
    # Specific attenuation coefficient (dB/m); ITU-R P.833-9 Eq. 2
    gamma = 0.39 * (frequency_mhz ** 0.39)


    # Typical values: 15-40 dB for dense vegetation
    if isinstance(canopy_height_m, torch.Tensor):
        A_m = 15.0 + 1.5 * canopy_height_m + 0.01 * frequency_mhz
        A_m = torch.clamp(A_m, min=5.0, max=60.0)


        if not isinstance(path_length_m, torch.Tensor):
            path_length = torch.full_like(canopy_height_m, float(path_length_m))
        else:
            path_length = path_length_m
        path_length = torch.clamp(path_length, min=0.0)


        attenuation = A_m * (1 - torch.exp(-path_length * gamma / A_m))


        no_veg_mask = canopy_height_m <= 0
        attenuation = torch.where(no_veg_mask, torch.zeros_like(attenuation), attenuation)

    else:
        canopy_height_m = np.asarray(canopy_height_m)
        A_m = 15.0 + 1.5 * canopy_height_m + 0.01 * frequency_mhz
        A_m = np.clip(A_m, 5.0, 60.0)

        path_length = np.clip(path_length_m, 0.0, None)
        attenuation = A_m * (1 - np.exp(-path_length * gamma / A_m))


        no_veg_mask = canopy_height_m <= 0
        attenuation = np.where(no_veg_mask, 0.0, attenuation)

    return attenuation


def vegetation_loss_ndvi_based(
    path_length_m: Union[float, np.ndarray],
    frequency_mhz: float,
    ndvi: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Computes vegetation attenuation using NDVI as a proxy.

    Empirical NDVI-based formula:
        A = k * NDVI * d * f^0.3

    Args:
        path_length_m: path length (m).
        frequency_mhz: frequency in MHz.
        ndvi: NDVI value (0-1; > 0.5 = dense vegetation).

    Returns:
        Estimated attenuation in dB.
    """

    ndvi = np.clip(ndvi, 0.0, 1.0)


    k = 0.05


    attenuation = k * ndvi * path_length_m * (frequency_mhz ** 0.3)


    return np.clip(attenuation, 0.0, 50.0)


def knife_edge_diffraction_loss(
    obstacle_height_m: Union[float, np.ndarray, torch.Tensor],
    d1_m: Union[float, np.ndarray, torch.Tensor],
    d2_m: Union[float, np.ndarray, torch.Tensor],
    frequency_mhz: float
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes diffraction loss with the Knife-Edge model.

    L_d(dB) = 6.9 + 20*log10(sqrt((v-0.1)^2 + 1) + v - 0.1)

    where v (Fresnel-Kirchhoff parameter):
        v = h * sqrt(2/lambda * (1/d1 + 1/d2))

    Args:
        obstacle_height_m: obstacle height above the LOS (m).
        d1_m: distance TX -> obstacle (m).
        d2_m: distance obstacle -> RX (m).
        frequency_mhz: frequency in MHz.

    Returns:
        Diffraction loss in dB (0 if LOS is clear).

    Example:
        >>> loss = knife_edge_diffraction_loss(50, 1000, 2000, 1800)
        >>> print(f"Diffraction loss: {loss:.2f} dB")
    """

    wavelength_m = SPEED_OF_LIGHT / (frequency_mhz * 1e6)

    if isinstance(obstacle_height_m, torch.Tensor):
        d1 = torch.clamp(d1_m, min=1.0)
        d2 = torch.clamp(d2_m, min=1.0)
        h = obstacle_height_m


        nu = h * torch.sqrt(2.0 / wavelength_m * (1.0/d1 + 1.0/d2))


        term = torch.sqrt((nu - 0.1)**2 + 1) + nu - 0.1
        loss = 6.9 + 20 * torch.log10(torch.clamp(term, min=0.001))


        loss = torch.where(h <= 0, torch.zeros_like(loss), loss)
        loss = torch.clamp(loss, min=0.0, max=40.0)

    else:
        d1 = np.clip(d1_m, 1.0, None)
        d2 = np.clip(d2_m, 1.0, None)
        h = np.asarray(obstacle_height_m)


        nu = h * np.sqrt(2.0 / wavelength_m * (1.0/d1 + 1.0/d2))


        term = np.sqrt((nu - 0.1)**2 + 1) + nu - 0.1
        loss = 6.9 + 20 * np.log10(np.clip(term, 0.001, None))

        # If the obstacle is below the LOS, loss = 0
        loss = np.where(h <= 0, 0.0, loss)
        loss = np.clip(loss, 0.0, 40.0)

    return loss


def fresnel_zone_radius(
    d1_m: float,
    d2_m: float,
    frequency_mhz: float,
    zone: int = 1
) -> float:
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
    wavelength_m = SPEED_OF_LIGHT / (frequency_mhz * 1e6)
    d_total = d1_m + d2_m

    if d_total < 1.0:
        return 0.0

    radius = np.sqrt(zone * wavelength_m * d1_m * d2_m / d_total)
    return radius


@dataclass
class PropagationResult:
    """Result of a propagation-loss computation."""
    total_loss_db: float
    fspl_db: float
    vegetation_db: float
    diffraction_db: float
    rssi_dbm: float
    distance_m: float
    is_los: bool


def total_path_loss(
    distance_m: Union[float, np.ndarray, torch.Tensor],
    frequency_mhz: float,
    canopy_height_m: Union[float, np.ndarray, torch.Tensor] = 0.0,
    obstacle_height_m: Union[float, np.ndarray, torch.Tensor] = 0.0,
    d1_ratio: float = 0.5,
    vegetation_path_m: Optional[Union[float, np.ndarray, torch.Tensor]] = None
) -> Tuple:
    """
    Computes total propagation loss combining all component models.

    L_total = L_fspl + L_vegetation + L_diffraction

    Args:
        distance_m: total TX -> RX distance (m).
        frequency_mhz: frequency in MHz.
        canopy_height_m: canopy height along the path (m).
        obstacle_height_m: obstacle height above the LOS (m).
        d1_ratio: ratio d1/(d1+d2) used for the diffraction term (default: 0.5).
        vegetation_path_m: path length through vegetation (if None, uses the total distance).

    Returns:
        Tuple: (total_loss, fspl, veg_loss, diff_loss).
    """

    fspl = free_space_path_loss(distance_m, frequency_mhz)


    if vegetation_path_m is None:
        vegetation_path_m = distance_m * 0.3  # Assumes 30% of the path is through vegetation
    veg_loss = vegetation_loss_itu_p833(vegetation_path_m, frequency_mhz, canopy_height_m)


    if isinstance(distance_m, torch.Tensor):
        d1 = distance_m * d1_ratio
        d2 = distance_m * (1 - d1_ratio)
    else:
        d1 = np.asarray(distance_m) * d1_ratio
        d2 = np.asarray(distance_m) * (1 - d1_ratio)

    diff_loss = knife_edge_diffraction_loss(obstacle_height_m, d1, d2, frequency_mhz)


    total = fspl + veg_loss + diff_loss

    return total, fspl, veg_loss, diff_loss


def calculate_rssi(
    distance_m: Union[float, np.ndarray, torch.Tensor],
    tx_power_dbm: float,
    frequency_mhz: float,
    canopy_height_m: Union[float, np.ndarray, torch.Tensor] = 0.0,
    obstacle_height_m: Union[float, np.ndarray, torch.Tensor] = 0.0,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0
) -> Tuple:
    """
    Computes RSSI accounting for all loss terms.

    RSSI = P_tx + G_tx + G_rx - L_total

    Args:
        distance_m: TX -> RX distance (m).
        tx_power_dbm: transmit power (dBm).
        frequency_mhz: frequency (MHz).
        canopy_height_m: canopy height (m).
        obstacle_height_m: obstacle height above the LOS (m).
        tx_gain_dbi: TX antenna gain (dBi).
        rx_gain_dbi: RX antenna gain (dBi).

    Returns:
        Tuple: (rssi_dbm, total_loss, fspl, veg_loss, diff_loss).
    """
    total_loss, fspl, veg_loss, diff_loss = total_path_loss(
        distance_m, frequency_mhz, canopy_height_m, obstacle_height_m
    )

    rssi = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_loss

    return rssi, total_loss, fspl, veg_loss, diff_loss


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTE: Propagation Models")
    print("=" * 70)


    freq_mhz = 1800
    tx_power = 43    # dBm (typical eNodeB)

    print("\n## 1. Free Space Path Loss (Friis)")
    print("-" * 50)
    for d in [100, 500, 1000, 5000, 10000]:
        fspl = free_space_path_loss(d, freq_mhz)
        print(f"   {d:>5}m: {fspl:.2f} dB")

    print("\n## 2. ITU-R P.833 (Vegetação)")
    print("-" * 50)
    for canopy in [0, 5, 10, 15, 20]:
        veg = vegetation_loss_itu_p833(100, freq_mhz, canopy)
        print(f"   Canopy {canopy}m, 100m path: {veg:.2f} dB")

    print("\n## 3. Knife-Edge Diffraction")
    print("-" * 50)
    for h in [-10, 0, 10, 30, 50]:
        diff = knife_edge_diffraction_loss(h, 1000, 2000, freq_mhz)
        print(f"   Obstacle {h:>3}m: {diff:.2f} dB")

    print("\n## 4. RSSI Calculation")
    print("-" * 50)
    for d in [500, 1000, 2000, 5000]:
        rssi, total, fspl, veg, diff = calculate_rssi(
            d, tx_power, freq_mhz,
            canopy_height_m=10,
            obstacle_height_m=0,
            tx_gain_dbi=15
        )
        print(f"   {d:>5}m: RSSI={rssi:.1f} dBm (Loss: FSPL={fspl:.1f}, Veg={veg:.1f}, Diff={diff:.1f})")

    print("\n## 5. Fresnel Zone Radius")
    print("-" * 50)
    for d in [1000, 5000, 10000]:
        r1 = fresnel_zone_radius(d/2, d/2, freq_mhz, zone=1)
        print(f"   {d:>5}m total, midpoint: r1={r1:.2f}m")

    print("\n## 6. Tensor Support Test")
    print("-" * 50)
    distances = torch.tensor([100, 500, 1000, 2000, 5000], dtype=torch.float32)
    fspl_tensor = free_space_path_loss(distances, freq_mhz)
    print(f"   Tensor FSPL: {fspl_tensor.numpy().round(2)}")

    canopy = torch.tensor([0, 5, 10, 15, 20], dtype=torch.float32)
    veg_tensor = vegetation_loss_itu_p833(100.0, freq_mhz, canopy)
    print(f"   Tensor Veg:  {veg_tensor.numpy().round(2)}")

    print("\n" + "=" * 70)
    print("✅ TESTES CONCLUÍDOS")
    print("=" * 70)
