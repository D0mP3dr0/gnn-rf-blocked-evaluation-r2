# GNN-RF Propagation V2
# 02_models/los_calculator.py

"""
Cálculo de Linha de Visada (LOS) para propagação RF.

Funcionalidades:
    - Interpolação de path entre TX e RX
    - Verificação de obstrução considerando DEM + Canopy
    - Cálculo de clearance da zona de Fresnel
    - Identificação de pontos de difração

Referências:
    - ITU-R P.526-15: Propagation by diffraction
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import math


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class LOSResult:
    """Resultado da análise de linha de visada."""
    is_los: bool                    # True se há linha de visada clara
    obstruction_height_m: float     # Altura da maior obstrução acima da LOS
    fresnel_clearance_pct: float    # % de clearance da 1ª zona Fresnel (>60% = good)
    obstruction_distance_m: float   # Distância TX → ponto de maior obstrução
    num_obstructions: int           # Número de obstruções
    path_length_m: float            # Comprimento total do path


@dataclass
class PathProfile:
    """Perfil de caminho entre TX e RX."""
    distances_m: np.ndarray        # Distância de cada ponto ao TX
    elevations_m: np.ndarray       # Elevação do terreno (DEM)
    canopy_m: np.ndarray           # Altura do dossel
    surface_m: np.ndarray          # DEM + Canopy
    los_heights_m: np.ndarray      # Altura da LOS em cada ponto
    clearances_m: np.ndarray       # LOS height - surface height
    fresnel_radii_m: np.ndarray    # Raio da 1ª zona Fresnel


# =============================================================================
# CONSTANTES
# =============================================================================

SPEED_OF_LIGHT = 299792458.0  # m/s
EARTH_RADIUS = 6371000.0      # m (raio médio da Terra)


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

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
    wavelength_m = calculate_wavelength(frequency_mhz)
    d_total = d1_m + d2_m
    
    # Evitar divisão por zero
    d_total = np.maximum(d_total, 1.0)
    
    radius = np.sqrt(zone * wavelength_m * d1_m * d2_m / d_total)
    return radius


def earth_curvature_correction(distance_m: Union[float, np.ndarray], k: float = 4/3) -> Union[float, np.ndarray]:
    """
    Calcula correção de curvatura da Terra.
    
    h_correction = d² / (2 · k · R)
    
    Args:
        distance_m: Distância do ponto de referência (m)
        k: Fator de curvatura efetiva (4/3 para atmosfera padrão)
    
    Returns:
        Correção de altura em metros (adicionar à elevação)
    """
    return (distance_m ** 2) / (2 * k * EARTH_RADIUS)


# =============================================================================
# INTERPOLAÇÃO DE PATH
# =============================================================================

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
    Interpola perfil de elevação ao longo do caminho TX-RX.
    
    Args:
        tx_pos: (lon, lat, altura_m) do transmissor
        rx_pos: (lon, lat, altura_m) do receptor
        dem_grid: Grid de elevação (DEM) [H, W]
        canopy_grid: Grid de altura do dossel [H, W]
        transform: Affine transform do raster
        num_points: Número de pontos de amostragem
        frequency_mhz: Frequência para cálculo de Fresnel
        apply_earth_curvature: Aplicar correção de curvatura da Terra
    
    Returns:
        PathProfile com todos os dados do caminho
    """
    tx_lon, tx_lat, tx_height = tx_pos
    rx_lon, rx_lat, rx_height = rx_pos
    
    # Gerar pontos ao longo do path
    t = np.linspace(0, 1, num_points)
    lons = tx_lon + t * (rx_lon - tx_lon)
    lats = tx_lat + t * (rx_lat - tx_lat)
    
    # Calcular distâncias (aproximação planar para distâncias curtas)
    # Para distâncias maiores, usar haversine
    dx = (rx_lon - tx_lon) * 111320 * np.cos(np.radians((tx_lat + rx_lat) / 2))
    dy = (rx_lat - tx_lat) * 110540
    total_distance = np.sqrt(dx**2 + dy**2)
    distances = t * total_distance
    
    # Amostrar DEM e Canopy
    # Converter coordenadas geográficas para índices de pixel
    elevations = np.zeros(num_points)
    canopy = np.zeros(num_points)
    
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        # Converter coordenada para índice de pixel
        col = int((lon - transform.c) / transform.a)
        row = int((lat - transform.f) / transform.e)
        
        # Verificar limites
        if 0 <= row < dem_grid.shape[0] and 0 <= col < dem_grid.shape[1]:
            elevations[i] = dem_grid[row, col]
            canopy[i] = canopy_grid[row, col] if canopy_grid is not None else 0
        else:
            elevations[i] = 0
            canopy[i] = 0
    
    # Superfície = DEM + Canopy
    surface = elevations + canopy
    
    # Aplicar correção de curvatura da Terra
    if apply_earth_curvature:
        # Correção relativa ao ponto médio
        mid_distance = total_distance / 2
        curvature = earth_curvature_correction(np.abs(distances - mid_distance))
        surface = surface + curvature
    
    # Calcular altura da LOS em cada ponto
    los_heights = tx_height + t * (rx_height - tx_height)
    
    # Adicionar elevação do terreno nos pontos TX e RX à LOS
    # LOS é uma linha reta entre (0, tx_elev + tx_height) e (total_dist, rx_elev + rx_height)
    tx_total = elevations[0] + tx_height
    rx_total = elevations[-1] + rx_height
    los_heights = tx_total + t * (rx_total - tx_total)
    
    # Calcular clearances
    clearances = los_heights - surface
    
    # Calcular raios de Fresnel
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


# =============================================================================
# VERIFICAÇÃO DE LOS
# =============================================================================

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
    Verifica se há linha de visada entre TX e RX.
    
    Args:
        tx_pos: (lon, lat, altura_m) do transmissor
        rx_pos: (lon, lat, altura_m) do receptor
        dem_grid: Grid de elevação [H, W]
        canopy_grid: Grid de altura do dossel [H, W] (opcional)
        transform: Affine transform do raster
        frequency_mhz: Frequência para cálculo de Fresnel
        num_points: Pontos de amostragem ao longo do path
        fresnel_clearance_required: % mínimo de clearance (0.6 = 60%)
    
    Returns:
        LOSResult com informações detalhadas
    """
    if canopy_grid is None:
        canopy_grid = np.zeros_like(dem_grid)
    
    if transform is None:
        # Criar transform padrão se não fornecido
        from rasterio.transform import Affine
        transform = Affine(0.0003, 0, tx_pos[0] - 0.01, 0, -0.0003, tx_pos[1] + 0.01)
    
    # Obter perfil do path
    profile = interpolate_path_profile(
        tx_pos, rx_pos, dem_grid, canopy_grid,
        transform, num_points, frequency_mhz
    )
    
    # Analisar obstruções
    # Obstrução = surface acima de (LOS - 60% do raio de Fresnel)
    required_clearance = fresnel_clearance_required * profile.fresnel_radii_m
    effective_clearance = profile.clearances_m - required_clearance
    
    # Ignorar pontos inicial e final (TX e RX)
    check_range = slice(1, -1)  # Excluir primeiro e último ponto
    
    obstructions = effective_clearance[check_range] < 0
    num_obstructions = np.sum(obstructions)
    
    if num_obstructions > 0:
        # Encontrar maior obstrução
        obstruction_indices = np.where(obstructions)[0] + 1  # +1 pelo slice
        obstruction_heights = -effective_clearance[check_range][obstructions]
        max_obstruction_idx = obstruction_indices[np.argmax(obstruction_heights)]
        
        max_obstruction_height = np.max(obstruction_heights)
        obstruction_distance = profile.distances_m[max_obstruction_idx]
        
        # Calcular clearance mínimo como percentual
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
        # LOS clara
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


# =============================================================================
# ANÁLISE DE LOS EM BATCH (para GNN)
# =============================================================================

def check_los_batch(
    tx_pos: Tuple[float, float, float],
    rx_positions: torch.Tensor,
    dem_tensor: torch.Tensor,
    canopy_tensor: Optional[torch.Tensor] = None,
    cell_size_m: float = 30.0,
    frequency_mhz: float = 1800.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Verifica LOS para múltiplos receptores em batch (otimizado para GNN).
    
    Args:
        tx_pos: (x, y, altura_m) do transmissor
        rx_positions: [N, 2] ou [N, 3] posições dos receptores
        dem_tensor: [H, W] elevação do terreno
        canopy_tensor: [H, W] altura do dossel (opcional)
        cell_size_m: Tamanho da célula em metros
        frequency_mhz: Frequência em MHz
    
    Returns:
        is_los: [N] bool tensor
        obstruction_height: [N] altura da obstrução em metros
    """
    device = rx_positions.device
    n_rx = rx_positions.shape[0]
    
    tx_x, tx_y, tx_h = tx_pos
    
    if canopy_tensor is None:
        canopy_tensor = torch.zeros_like(dem_tensor)
    
    # Surface = DEM + Canopy
    surface = dem_tensor + canopy_tensor
    
    is_los = torch.ones(n_rx, dtype=torch.bool, device=device)
    obstruction_heights = torch.zeros(n_rx, device=device)
    
    # Para cada receptor
    for i in range(n_rx):
        rx_x = rx_positions[i, 0].item()
        rx_y = rx_positions[i, 1].item()
        rx_h = rx_positions[i, 2].item() if rx_positions.shape[1] > 2 else 1.5  # altura padrão 1.5m
        
        # Distância
        dx = rx_x - tx_x
        dy = rx_y - tx_y
        distance = math.sqrt(dx**2 + dy**2) * cell_size_m
        
        if distance < cell_size_m:
            continue  # Muito perto, considera LOS
        
        # Amostrar pontos ao longo do path
        num_samples = int(distance / cell_size_m) + 1
        t = torch.linspace(0, 1, num_samples, device=device)
        
        sample_x = tx_x + t * dx
        sample_y = tx_y + t * dy
        
        # Converter para índices de grid
        col_idx = sample_x.long()
        row_idx = sample_y.long()
        
        # Verificar limites
        valid = (row_idx >= 0) & (row_idx < surface.shape[0]) & \
                (col_idx >= 0) & (col_idx < surface.shape[1])
        
        if not valid.any():
            continue
        
        # Amostrar alturas da superfície
        row_idx = torch.clamp(row_idx, 0, surface.shape[0] - 1)
        col_idx = torch.clamp(col_idx, 0, surface.shape[1] - 1)
        surface_heights = surface[row_idx, col_idx]
        
        # Calcular altura da LOS em cada ponto
        tx_total_h = surface[int(tx_y), int(tx_x)].item() + tx_h
        rx_total_h = surface[int(rx_y), int(rx_x)].item() + rx_h
        los_heights = tx_total_h + t * (rx_total_h - tx_total_h)
        
        # Verificar obstruções (excluindo pontos inicial e final)
        if num_samples > 2:
            clearances = los_heights[1:-1] - surface_heights[1:-1]
            min_clearance = clearances.min()
            
            if min_clearance < 0:
                is_los[i] = False
                obstruction_heights[i] = -min_clearance
    
    return is_los, obstruction_heights


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTE: LOS Calculator")
    print("=" * 70)
    
    # Criar terreno sintético
    np.random.seed(42)
    H, W = 200, 200
    
    # DEM com uma colina no meio
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    dem = 100 + 50 * np.exp(-(X**2 + Y**2) / 0.2)  # Colina gaussiana
    
    # Canopy variável
    canopy = np.random.uniform(0, 15, (H, W))
    canopy[dem > 130] = 0  # Sem árvores no topo da colina
    
    print("\n## 1. Teste de Interpolação de Path")
    print("-" * 50)
    
    from rasterio.transform import Affine
    transform = Affine(0.0003, 0, -47.0, 0, -0.0003, -22.0)
    
    tx_pos = (-47.0, -22.0, 30)  # Torre 30m
    rx_pos = (-46.95, -22.0, 1.5)  # Receptor 1.5m
    
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
    
    # Caso 1: LOS clara (antena alta, sem obstrução)
    result1 = check_line_of_sight(
        (-47.0, -22.0, 50),  # Torre 50m
        (-46.95, -22.0, 1.5),
        dem, canopy, transform
    )
    print(f"   Caso 1 (Torre 50m): LOS={result1.is_los}, Fresnel={result1.fresnel_clearance_pct:.1f}%")
    
    # Caso 2: LOS bloqueada (antena baixa, colina no meio)
    result2 = check_line_of_sight(
        (-47.02, -22.0, 5),   # Torre baixa
        (-46.98, -22.0, 1.5),  # Passa sobre a colina
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
    
    # Criar tensors
    dem_t = torch.tensor(dem, dtype=torch.float32)
    canopy_t = torch.tensor(canopy, dtype=torch.float32)
    
    # Múltiplos receptores
    rx_positions = torch.tensor([
        [150, 100, 1.5],  # Longe
        [110, 100, 1.5],  # Perto do topo da colina
        [50, 100, 1.5],   # Outro lado
    ], dtype=torch.float32)
    
    is_los, obst_h = check_los_batch(
        (100, 100, 30), rx_positions, dem_t, canopy_t, cell_size_m=30
    )
    print(f"   is_los: {is_los.numpy()}")
    print(f"   obstruction: {obst_h.numpy().round(2)}")
    
    print("\n" + "=" * 70)
    print("✅ TESTES CONCLUÍDOS")
    print("=" * 70)
