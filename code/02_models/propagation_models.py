# GNN-RF Propagation V2
# 02_models/propagation_models.py

"""
Modelos físicos de propagação RF.

Implementa:
    - Free Space Path Loss (Friis)
    - ITU-R P.833-9 (Vegetação)
    - Knife-Edge Diffraction (Deygout)

Referências:
    - ITU-R P.525-4: Free Space Propagation
    - ITU-R P.833-9: Attenuation in Vegetation
    - ITU-R P.526-15: Diffraction
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# CONSTANTES
# =============================================================================

SPEED_OF_LIGHT = 299792458.0  # m/s
DEFAULT_RSSI_THRESHOLD = -120.0  # dBm


# =============================================================================
# FREE SPACE PATH LOSS (Friis)
# =============================================================================

def free_space_path_loss(
    distance_m: Union[float, np.ndarray, torch.Tensor],
    frequency_mhz: float
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calcula Free Space Path Loss (FSPL) usando equação de Friis.
    
    FSPL(dB) = 20·log₁₀(d_km) + 20·log₁₀(f_MHz) + 32.44
    
    Args:
        distance_m: Distância em metros (pode ser array/tensor)
        frequency_mhz: Frequência em MHz
    
    Returns:
        Path loss em dB
    
    Example:
        >>> fspl = free_space_path_loss(1000, 1800)  # 1km, 1800 MHz
        >>> print(f"FSPL: {fspl:.2f} dB")  # ~97.5 dB
    """
    # Converter para km
    if isinstance(distance_m, torch.Tensor):
        distance_km = distance_m / 1000.0
        distance_km = torch.clamp(distance_km, min=0.001)  # Min 1m
        return 20 * torch.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.44
    else:
        distance_km = np.asarray(distance_m) / 1000.0
        distance_km = np.clip(distance_km, 0.001, None)  # Min 1m
        return 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.44


def fspl_with_gains(
    distance_m: float,
    frequency_mhz: float,
    tx_power_dbm: float,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0
) -> float:
    """
    Calcula potência recebida considerando ganhos de antena.
    
    P_rx = P_tx + G_tx + G_rx - FSPL
    
    Args:
        distance_m: Distância em metros
        frequency_mhz: Frequência em MHz
        tx_power_dbm: Potência de transmissão em dBm
        tx_gain_dbi: Ganho da antena TX em dBi
        rx_gain_dbi: Ganho da antena RX em dBi
    
    Returns:
        Potência recebida em dBm
    """
    fspl = free_space_path_loss(distance_m, frequency_mhz)
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl


# =============================================================================
# ITU-R P.833-9 (VEGETAÇÃO)
# =============================================================================

def vegetation_loss_itu_p833(
    path_length_m: Union[float, np.ndarray, torch.Tensor],
    frequency_mhz: float,
    canopy_height_m: Union[float, np.ndarray, torch.Tensor] = 10.0
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calcula atenuação por vegetação usando ITU-R P.833-9.
    
    A = A_m · (1 - exp(-d·γ/A_m))
    
    Onde:
        γ = 0.39 · f^0.39  (coeficiente de atenuação específica, dB/m)
        A_m = atenuação máxima para vegetação muito profunda
    
    Args:
        path_length_m: Comprimento do caminho através da vegetação (m)
        frequency_mhz: Frequência em MHz
        canopy_height_m: Altura do dossel em metros (afeta A_m)
    
    Returns:
        Atenuação por vegetação em dB
    
    Example:
        >>> loss = vegetation_loss_itu_p833(100, 1800, 15)  # 100m path, 1800 MHz, 15m canopy
        >>> print(f"Veg loss: {loss:.2f} dB")
    """
    # Coeficiente de atenuação específica (dB/m) - ITU-R P.833-9 Eq. 2
    gamma = 0.39 * (frequency_mhz ** 0.39)
    
    # Atenuação máxima baseada na altura do dossel e frequência
    # Valores típicos: 15-40 dB para vegetação densa
    if isinstance(canopy_height_m, torch.Tensor):
        A_m = 15.0 + 1.5 * canopy_height_m + 0.01 * frequency_mhz
        A_m = torch.clamp(A_m, min=5.0, max=60.0)
        
        # Converter path_length para tensor se necessário
        if not isinstance(path_length_m, torch.Tensor):
            path_length = torch.full_like(canopy_height_m, float(path_length_m))
        else:
            path_length = path_length_m
        path_length = torch.clamp(path_length, min=0.0)
        
        # Atenuação calculada
        attenuation = A_m * (1 - torch.exp(-path_length * gamma / A_m))
        
        # Sem vegetação = sem perda
        no_veg_mask = canopy_height_m <= 0
        attenuation = torch.where(no_veg_mask, torch.zeros_like(attenuation), attenuation)
        
    else:
        canopy_height_m = np.asarray(canopy_height_m)
        A_m = 15.0 + 1.5 * canopy_height_m + 0.01 * frequency_mhz
        A_m = np.clip(A_m, 5.0, 60.0)
        
        path_length = np.clip(path_length_m, 0.0, None)
        attenuation = A_m * (1 - np.exp(-path_length * gamma / A_m))
        
        # Sem vegetação = sem perda
        no_veg_mask = canopy_height_m <= 0
        attenuation = np.where(no_veg_mask, 0.0, attenuation)
    
    return attenuation


def vegetation_loss_ndvi_based(
    path_length_m: Union[float, np.ndarray],
    frequency_mhz: float,
    ndvi: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calcula atenuação por vegetação usando NDVI como proxy.
    
    Fórmula empírica baseada em NDVI:
        A = k · NDVI · d · f^0.3
    
    Args:
        path_length_m: Comprimento do caminho (m)
        frequency_mhz: Frequência em MHz
        ndvi: Valor NDVI (0-1, onde >0.5 = vegetação densa)
    
    Returns:
        Atenuação estimada em dB
    """
    # NDVI -> densidade de vegetação
    ndvi = np.clip(ndvi, 0.0, 1.0)
    
    # Coeficiente empírico
    k = 0.05
    
    # Atenuação
    attenuation = k * ndvi * path_length_m * (frequency_mhz ** 0.3)
    
    # Saturação máxima
    return np.clip(attenuation, 0.0, 50.0)


# =============================================================================
# KNIFE-EDGE DIFFRACTION (Deygout)
# =============================================================================

def knife_edge_diffraction_loss(
    obstacle_height_m: Union[float, np.ndarray, torch.Tensor],
    d1_m: Union[float, np.ndarray, torch.Tensor],
    d2_m: Union[float, np.ndarray, torch.Tensor],
    frequency_mhz: float
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calcula perda por difração usando modelo Knife-Edge.
    
    L_d(dB) = 6.9 + 20·log₁₀(√((ν-0.1)² + 1) + ν - 0.1)
    
    Onde ν (parâmetro de Fresnel-Kirchhoff):
        ν = h · √(2/λ · (1/d₁ + 1/d₂))
    
    Args:
        obstacle_height_m: Altura do obstáculo acima da LOS (m)
        d1_m: Distância TX → obstáculo (m)
        d2_m: Distância obstáculo → RX (m)
        frequency_mhz: Frequência em MHz
    
    Returns:
        Perda por difração em dB (0 se LOS clara)
    
    Example:
        >>> loss = knife_edge_diffraction_loss(50, 1000, 2000, 1800)
        >>> print(f"Diffraction loss: {loss:.2f} dB")
    """
    # Comprimento de onda
    wavelength_m = SPEED_OF_LIGHT / (frequency_mhz * 1e6)
    
    if isinstance(obstacle_height_m, torch.Tensor):
        d1 = torch.clamp(d1_m, min=1.0)
        d2 = torch.clamp(d2_m, min=1.0)
        h = obstacle_height_m
        
        # Parâmetro de Fresnel-Kirchhoff
        nu = h * torch.sqrt(2.0 / wavelength_m * (1.0/d1 + 1.0/d2))
        
        # Perda por difração (ITU-R P.526-15)
        # Para ν > -0.7 (obstáculo acima da LOS)
        term = torch.sqrt((nu - 0.1)**2 + 1) + nu - 0.1
        loss = 6.9 + 20 * torch.log10(torch.clamp(term, min=0.001))
        
        # Se obstáculo abaixo da LOS (h < 0), perda = 0
        loss = torch.where(h <= 0, torch.zeros_like(loss), loss)
        loss = torch.clamp(loss, min=0.0, max=40.0)
        
    else:
        d1 = np.clip(d1_m, 1.0, None)
        d2 = np.clip(d2_m, 1.0, None)
        h = np.asarray(obstacle_height_m)
        
        # Parâmetro de Fresnel-Kirchhoff
        nu = h * np.sqrt(2.0 / wavelength_m * (1.0/d1 + 1.0/d2))
        
        # Perda por difração
        term = np.sqrt((nu - 0.1)**2 + 1) + nu - 0.1
        loss = 6.9 + 20 * np.log10(np.clip(term, 0.001, None))
        
        # Se obstáculo abaixo da LOS, perda = 0
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
    Calcula o raio da zona de Fresnel.
    
    r_n = √(n · λ · d₁ · d₂ / (d₁ + d₂))
    
    Args:
        d1_m: Distância TX → ponto (m)
        d2_m: Distância ponto → RX (m)
        frequency_mhz: Frequência em MHz
        zone: Número da zona de Fresnel (default: 1)
    
    Returns:
        Raio da zona de Fresnel em metros
    """
    wavelength_m = SPEED_OF_LIGHT / (frequency_mhz * 1e6)
    d_total = d1_m + d2_m
    
    if d_total < 1.0:
        return 0.0
    
    radius = np.sqrt(zone * wavelength_m * d1_m * d2_m / d_total)
    return radius


# =============================================================================
# MODELO COMBINADO
# =============================================================================

@dataclass
class PropagationResult:
    """Resultado do cálculo de propagação."""
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
    Calcula perda total de propagação combinando todos os modelos.
    
    L_total = L_fspl + L_vegetation + L_diffraction
    
    Args:
        distance_m: Distância total TX → RX (m)
        frequency_mhz: Frequência em MHz
        canopy_height_m: Altura do dossel no caminho (m)
        obstacle_height_m: Altura do obstáculo acima da LOS (m)
        d1_ratio: Razão d1/(d1+d2) para cálculo de difração (default: 0.5)
        vegetation_path_m: Comprimento do caminho através da vegetação (se None, usa distância total)
    
    Returns:
        Tuple: (total_loss, fspl, veg_loss, diff_loss)
    """
    # 1. Free Space Path Loss
    fspl = free_space_path_loss(distance_m, frequency_mhz)
    
    # 2. Vegetation Loss
    if vegetation_path_m is None:
        vegetation_path_m = distance_m * 0.3  # Assume 30% do caminho em vegetação
    veg_loss = vegetation_loss_itu_p833(vegetation_path_m, frequency_mhz, canopy_height_m)
    
    # 3. Diffraction Loss
    if isinstance(distance_m, torch.Tensor):
        d1 = distance_m * d1_ratio
        d2 = distance_m * (1 - d1_ratio)
    else:
        d1 = np.asarray(distance_m) * d1_ratio
        d2 = np.asarray(distance_m) * (1 - d1_ratio)
    
    diff_loss = knife_edge_diffraction_loss(obstacle_height_m, d1, d2, frequency_mhz)
    
    # Total
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
    Calcula RSSI considerando todos os fatores de perda.
    
    RSSI = P_tx + G_tx + G_rx - L_total
    
    Args:
        distance_m: Distância TX → RX (m)
        tx_power_dbm: Potência de transmissão (dBm)
        frequency_mhz: Frequência (MHz)
        canopy_height_m: Altura do dossel (m)
        obstacle_height_m: Altura do obstáculo acima da LOS (m)
        tx_gain_dbi: Ganho da antena TX (dBi)
        rx_gain_dbi: Ganho da antena RX (dBi)
    
    Returns:
        Tuple: (rssi_dbm, total_loss, fspl, veg_loss, diff_loss)
    """
    total_loss, fspl, veg_loss, diff_loss = total_path_loss(
        distance_m, frequency_mhz, canopy_height_m, obstacle_height_m
    )
    
    rssi = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_loss
    
    return rssi, total_loss, fspl, veg_loss, diff_loss


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTE: Propagation Models")
    print("=" * 70)
    
    # Parâmetros de teste
    freq_mhz = 1800  # LTE Band 3
    tx_power = 43    # dBm (típico eNodeB)
    
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
