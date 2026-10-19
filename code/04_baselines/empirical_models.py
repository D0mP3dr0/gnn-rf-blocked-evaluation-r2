# GNN-RF Propagation V2
# 04_baselines/empirical_models.py

"""
Modelos empíricos de propagação RF para comparação.

Implementa:
    - Free Space Path Loss (FSPL)
    - Okumura-Hata (urbano/suburbano/rural)
    - COST-231 Hata
    - Log-distance path loss
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


def free_space_path_loss(distance_m: np.ndarray,
                        frequency_mhz: float) -> np.ndarray:
    """
    Free Space Path Loss (Friis).
    
    FSPL = 20*log10(d) + 20*log10(f) + 32.44
    
    Args:
        distance_m: Distância em metros
        frequency_mhz: Frequência em MHz
    
    Returns:
        Path loss em dB
    """
    d_km = np.maximum(distance_m / 1000.0, 0.001)
    fspl = 20 * np.log10(d_km) + 20 * np.log10(frequency_mhz) + 32.44
    return fspl


def okumura_hata_urban(distance_m: np.ndarray,
                       frequency_mhz: float,
                       h_tx: float = 30.0,
                       h_rx: float = 1.5) -> np.ndarray:
    """
    Modelo Okumura-Hata para ambiente urbano.
    
    Válido para: 150-1500 MHz, 1-20km
    
    Args:
        distance_m: Distância em metros
        frequency_mhz: Frequência em MHz
        h_tx: Altura da antena transmissora (m)
        h_rx: Altura do receptor (m)
    
    Returns:
        Path loss em dB
    """
    f = frequency_mhz
    d = np.maximum(distance_m / 1000.0, 0.001)  # km
    
    # Fator de correção para altura do receptor (cidade grande)
    if f <= 300:
        a_hr = 8.29 * (np.log10(1.54 * h_rx))**2 - 1.1
    else:
        a_hr = 3.2 * (np.log10(11.75 * h_rx))**2 - 4.97
    
    # Path loss urbano
    L = (69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(h_tx) - a_hr +
         (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d))
    
    return L


def okumura_hata_suburban(distance_m: np.ndarray,
                          frequency_mhz: float,
                          h_tx: float = 30.0,
                          h_rx: float = 1.5) -> np.ndarray:
    """
    Modelo Okumura-Hata para ambiente suburbano.
    """
    L_urban = okumura_hata_urban(distance_m, frequency_mhz, h_tx, h_rx)
    f = frequency_mhz
    
    # Correção para subúrbio
    L = L_urban - 2 * (np.log10(f / 28))**2 - 5.4
    
    return L


def okumura_hata_rural(distance_m: np.ndarray,
                       frequency_mhz: float,
                       h_tx: float = 30.0,
                       h_rx: float = 1.5) -> np.ndarray:
    """
    Modelo Okumura-Hata para ambiente rural (open area).
    """
    L_urban = okumura_hata_urban(distance_m, frequency_mhz, h_tx, h_rx)
    f = frequency_mhz
    
    # Correção para área rural
    L = L_urban - 4.78 * (np.log10(f))**2 + 18.33 * np.log10(f) - 40.94
    
    return L


def cost231_hata(distance_m: np.ndarray,
                 frequency_mhz: float,
                 h_tx: float = 30.0,
                 h_rx: float = 1.5,
                 environment: str = 'urban') -> np.ndarray:
    """
    Modelo COST-231 Hata (extensão para 1500-2000 MHz).
    
    Args:
        distance_m: Distância em metros
        frequency_mhz: Frequência em MHz (1500-2000)
        h_tx: Altura da antena transmissora (m)
        h_rx: Altura do receptor (m)
        environment: 'urban' ou 'suburban'
    
    Returns:
        Path loss em dB
    """
    f = frequency_mhz
    d = np.maximum(distance_m / 1000.0, 0.001)  # km
    
    # Fator de correção para altura do receptor
    a_hr = (1.1 * np.log10(f) - 0.7) * h_rx - (1.56 * np.log10(f) - 0.8)
    
    # Constante de ambiente
    C_m = 3.0 if environment == 'urban' else 0.0
    
    # Path loss
    L = (46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_tx) - a_hr +
         (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d) + C_m)
    
    return L


def log_distance_path_loss(distance_m: np.ndarray,
                          frequency_mhz: float,
                          n: float = 3.5,
                          d0: float = 1.0,
                          X_sigma: float = 0.0) -> np.ndarray:
    """
    Modelo log-distance genérico.
    
    PL = PL(d0) + 10*n*log10(d/d0) + X_sigma
    
    Args:
        distance_m: Distância em metros
        frequency_mhz: Frequência em MHz
        n: Expoente de path loss (2-6)
        d0: Distância de referência (m)
        X_sigma: Variação de shadowing (dB)
    
    Returns:
        Path loss em dB
    """
    d = np.maximum(distance_m, d0)
    
    # FSPL na distância de referência
    PL_d0 = free_space_path_loss(np.array([d0]), frequency_mhz)[0]
    
    # Log-distance
    PL = PL_d0 + 10 * n * np.log10(d / d0) + X_sigma
    
    return PL


class EmpiricalBaseline:
    """
    Wrapper para usar modelos empíricos como baseline.
    """
    
    MODELS = {
        'fspl': free_space_path_loss,
        'okumura_hata_urban': okumura_hata_urban,
        'okumura_hata_suburban': okumura_hata_suburban,
        'okumura_hata_rural': okumura_hata_rural,
        'cost231_urban': lambda d, f: cost231_hata(d, f, environment='urban'),
        'cost231_suburban': lambda d, f: cost231_hata(d, f, environment='suburban'),
        'log_distance': log_distance_path_loss
    }
    
    def __init__(self, model_name: str = 'fspl'):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(self.MODELS.keys())}")
        self.model_name = model_name
        self.model_fn = self.MODELS[model_name]
    
    def predict(self,
               distances: np.ndarray,
               frequency_mhz: float,
               p_tx_dbm: float = 30.0,
               **kwargs) -> Dict[str, np.ndarray]:
        """
        Prediz path loss e RSSI.
        
        Returns:
            Dict com path_loss e rssi
        """
        # Path loss
        path_loss = self.model_fn(distances, frequency_mhz, **kwargs)
        
        # RSSI
        rssi = p_tx_dbm - path_loss
        
        # Coverage (threshold = -100 dBm)
        coverage = (rssi > -100).astype(float)
        
        return {
            'path_loss_total': path_loss,
            'rssi': rssi,
            'coverage_prob': coverage
        }


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: Modelos Empíricos")
    print("=" * 60)
    
    # Testar para diferentes distâncias
    distances = np.array([100, 500, 1000, 2000, 5000, 10000])  # metros
    frequency = 900.0  # MHz
    
    print(f"\n📊 Frequência: {frequency} MHz")
    print(f"   Distâncias: {distances} m")
    
    print("\n📋 Path Loss (dB):")
    print("-" * 80)
    print(f"{'Distância':<12}", end="")
    for name in ['fspl', 'okumura_hata_urban', 'okumura_hata_suburban', 'cost231_urban']:
        print(f"{name:<20}", end="")
    print()
    print("-" * 80)
    
    for d in distances:
        print(f"{d:<12.0f}", end="")
        for name in ['fspl', 'okumura_hata_urban', 'okumura_hata_suburban', 'cost231_urban']:
            baseline = EmpiricalBaseline(name)
            result = baseline.predict(np.array([d]), frequency)
            print(f"{result['path_loss_total'][0]:<20.1f}", end="")
        print()
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO")
    print("=" * 60)
