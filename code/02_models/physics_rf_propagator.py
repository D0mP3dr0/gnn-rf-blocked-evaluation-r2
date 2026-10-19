# GNN-RF Propagation V2
# 02_models/physics_rf_propagator.py

"""
Physics-Based RF Propagator usando Message Passing.

Propaga sinal RF através do grafo usando física real:
    - Free Space Path Loss
    - Vegetação (ITU-R P.833)
    - Difração (Knife-Edge)
    - LOS/NLOS detection

O sinal é propagado iterativamente dos nós de antena para os nós de terreno,
calculando RSSI em cada nó atingido.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Importar modelos de propagação
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from propagation_models import (
    free_space_path_loss,
    vegetation_loss_itu_p833,
    knife_edge_diffraction_loss,
    calculate_rssi,
    SPEED_OF_LIGHT
)


# =============================================================================
# CONSTANTES
# =============================================================================

DEFAULT_FREQUENCY_MHZ = 1800.0
DEFAULT_TX_POWER_DBM = 43.0
DEFAULT_TX_GAIN_DBI = 15.0
DEFAULT_RX_GAIN_DBI = 0.0
DEFAULT_RSSI_THRESHOLD = -120.0
DEFAULT_CELL_SIZE_M = 30.0


# =============================================================================
# PHYSICS RF MESSAGE LAYER
# =============================================================================

class PhysicsRFMessage(nn.Module):
    """
    Camada de propagação RF com física real.
    
    Propaga o sinal RF de nós de antena para nós de terreno,
    aplicando atenuação física baseada em:
    - Distância (FSPL)
    - Vegetação (canopy height)
    - Difração (obstruction height)
    
    Usa scatter_max para agregar mensagens.
    
    Args:
        frequency_mhz: Frequência de operação
        cell_size_m: Tamanho da célula em metros
        rssi_threshold: Limiar mínimo de RSSI para continuar propagação
    """
    
    def __init__(
        self,
        frequency_mhz: float = DEFAULT_FREQUENCY_MHZ,
        cell_size_m: float = DEFAULT_CELL_SIZE_M,
        rssi_threshold: float = DEFAULT_RSSI_THRESHOLD,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        tx_gain_dbi: float = DEFAULT_TX_GAIN_DBI
    ):
        super().__init__()
        
        self.frequency_mhz = frequency_mhz
        self.cell_size_m = cell_size_m
        self.rssi_threshold = rssi_threshold
        self.tx_power_dbm = tx_power_dbm
        self.tx_gain_dbi = tx_gain_dbi
        self.wavelength_m = SPEED_OF_LIGHT / (frequency_mhz * 1e6)
    
    def forward(
        self,
        rssi: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        canopy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propaga sinal RF através das arestas.
        
        Args:
            rssi: [N] RSSI atual em cada nó (dBm)
            edge_index: [2, E] índices de arestas
            edge_attr: [E, D] atributos (distância, etc.)
            canopy: [N] altura do dossel em cada nó (opcional)
        
        Returns:
            rssi_updated: [N] RSSI atualizado após propagação
        """
        src, dst = edge_index
        n_nodes = rssi.shape[0]
        device = rssi.device
        
        if canopy is None:
            canopy = torch.zeros(n_nodes, device=device)
        
        # Extrair distância das arestas
        if edge_attr.dim() == 1:
            distance_cells = edge_attr
        else:
            distance_cells = edge_attr[:, 0]
        
        distance_m = distance_cells * self.cell_size_m
        distance_m = torch.clamp(distance_m, min=self.cell_size_m)
        
        # RSSI no nó de origem
        rssi_src = rssi[src]
        
        # 2. Vegetation Loss (Linear)
        canopy_src = canopy[src]
        canopy_dst = canopy[dst]
        avg_canopy = (canopy_src + canopy_dst) / 2
        veg_path_m = distance_m * 0.3
        veg_loss = vegetation_loss_itu_p833(veg_path_m, self.frequency_mhz, avg_canopy)
        
        # 3. Difração (se houver)
        if edge_attr.dim() > 1 and edge_attr.shape[1] > 1:
            elev_diff = edge_attr[:, 1]
            obstruction = torch.clamp(elev_diff, min=0)
            d1 = distance_m / 2
            d2 = distance_m / 2
            diff_loss = knife_edge_diffraction_loss(obstruction, d1, d2, self.frequency_mhz)
        else:
            diff_loss = torch.zeros_like(veg_loss)
            
        # Total loss e RSSI recebido
        # CORREÇÃO FÍSICA: Usar "Virtual Distance" para calcular perda diferencial
        # Se usarmos FSPL direto na aresta (30m), aplicamos ~70dB de perda a cada hop.
        # O correto é: RSSI_new = RSSI_old - 20*log10((d + delta)/d) - LinearLosses
        
        # 1. Estimar distância atual da fonte (d_virt) baseada no RSSI atual
        # Inverse Friis: d = 10^((EIRP - RSSI - K) / 20)
        # K = 20*log10(f) - 27.55
        eirp = self.tx_power_dbm + self.tx_gain_dbi
        k_factor = 20 * torch.log10(torch.tensor(self.frequency_mhz, device=device)) - 27.55
        
        # Evitar log de infinito: clamp RSSI
        # Se RSSI é muito baixo, d_virt é enorme, perda geométrica tende a 0 (10*log(1) = 0)
        # Se RSSI é muito alto (perto da fonte), d_virt é pequeno, perda é alta
        rssi_clamped = torch.clamp(rssi_src, max=eirp - 1.0) # Evitar d=0
        
        val = (eirp - rssi_clamped - k_factor) / 20.0
        d_virt = torch.pow(10.0, val)
        
        # Evitar singularidades
        d_virt = torch.clamp(d_virt, min=1.0)
        
        # 2. Calcular nova distância
        d_new = d_virt + distance_m
        
        # 3. Calcular perda geométrica (Spreading Loss) diferencial
        # Loss_geo = 20 * log10(d_new / d_virt)
        geo_loss_db = 20 * torch.log10(d_new / d_virt)
        
        # 4. Somar perdas lineares (Vegetação + Difração)
        # Estas são perdas adicionais que acontecem NA ARESTA
        linear_losses = veg_loss + diff_loss
        
        # Total loss para este hop
        total_loss_hop = geo_loss_db + linear_losses
        
        # RSSI recebido
        rssi_received = rssi_src - total_loss_hop
        
        # DEBUG: mostrar estatísticas na primeira iteração
        # print(f"      FSPL: {fspl.mean():.1f}±{fspl.std():.1f} dB, Veg: {veg_loss.mean():.1f} dB")
        
        # Aplicar threshold - sinais abaixo não propagam
        valid_mask = rssi_received >= self.rssi_threshold
        
        # Agregar usando scatter_reduce (vetorizado, muito mais rápido)
        # Para cada nó destino, pegar o maior RSSI recebido
        rssi_new = torch.full((n_nodes,), float('-inf'), device=device)
        
        if valid_mask.any():
            # Usar scatter com reduce='max' para agregação eficiente
            valid_dst = dst[valid_mask]
            valid_rssi = rssi_received[valid_mask]
            
            # scatter_reduce com 'amax' (max reduction)
            rssi_new.scatter_reduce_(0, valid_dst, valid_rssi, reduce='amax', include_self=False)
        
        # Manter o maior entre atual e novo
        rssi_updated = torch.maximum(rssi, rssi_new)
        
        return rssi_updated


# =============================================================================
# PHYSICS RF PROPAGATOR
# =============================================================================

class PhysicsRFPropagator(nn.Module):
    """
    Propagador RF completo usando physics-based message passing.
    
    Propaga iterativamente o sinal das antenas para todo o terreno,
    usando física real para calcular atenuação.
    
    Args:
        frequency_mhz: Frequência de operação
        tx_power_dbm: Potência de transmissão
        tx_gain_dbi: Ganho da antena TX
        max_hops: Número máximo de hops de propagação
        cell_size_m: Tamanho da célula em metros
        rssi_threshold: Limiar mínimo de RSSI
    """
    
    def __init__(
        self,
        frequency_mhz: float = DEFAULT_FREQUENCY_MHZ,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        tx_gain_dbi: float = DEFAULT_TX_GAIN_DBI,
        max_hops: int = 10,
        cell_size_m: float = DEFAULT_CELL_SIZE_M,
        rssi_threshold: float = DEFAULT_RSSI_THRESHOLD
    ):
        super().__init__()
        
        self.frequency_mhz = frequency_mhz
        self.tx_power_dbm = tx_power_dbm
        self.tx_gain_dbi = tx_gain_dbi
        self.max_hops = max_hops
        self.cell_size_m = cell_size_m
        self.rssi_threshold = rssi_threshold
        
        # EIRP = Effective Isotropic Radiated Power
        self.eirp_dbm = tx_power_dbm + tx_gain_dbi
        
        # Camada de message passing
        self.rf_layer = PhysicsRFMessage(
            frequency_mhz=frequency_mhz,
            cell_size_m=cell_size_m,
            rssi_threshold=rssi_threshold,
            tx_power_dbm=tx_power_dbm,
            tx_gain_dbi=tx_gain_dbi
        )
    
    def forward(
        self,
        data: HeteroData,
        antenna_indices: Optional[torch.Tensor] = None,
        return_per_hop: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Propaga sinal RF de antenas para todo o terreno.
        
        Args:
            data: HeteroData com:
                - data['terrain'].x: Features do terreno [N, F]
                - data['terrain'].pos: Posições [N, 2]
                - data['antenna'].x: Features das antenas [M, F]
                - data['antenna'].pos: Posições das antenas [M, 2]
                - data[('terrain', 'connects_to', 'terrain')].edge_index: Arestas
            antenna_indices: Índices das antenas ativas (opcional)
            return_per_hop: Se True, retorna RSSI a cada hop
        
        Returns:
            Dict com:
                - 'rssi': [N] RSSI final em cada nó de terreno
                - 'path_loss': [N] Path loss total
                - 'coverage_mask': [N] Máscara de cobertura (>threshold)
                - 'hops_to_reach': [N] Número de hops para atingir cada nó
        """
        device = data['terrain'].x.device
        n_terrain = data['terrain'].x.shape[0]
        
        # Inicializar RSSI com -inf (não atingido)
        rssi = torch.full((n_terrain,), float('-inf'), device=device)
        hops_to_reach = torch.full((n_terrain,), -1, dtype=torch.int32, device=device)
        
        # Obter edge_index do terreno
        edge_key = ('terrain', 'connects_to', 'terrain')
        edge_index = None
        
        # Tentar diferentes padrões de edge types
        possible_keys = [
            ('terrain', 'connects_to', 'terrain'),
            ('terrain', 'to', 'terrain'),
        ]
        
        for key in possible_keys:
            if key in data.edge_types:
                candidate = data[key].edge_index
                # Validar que os índices estão dentro do range
                if candidate.max() < n_terrain:
                    edge_index = candidate
                    break
        
        # Se ainda não encontrou ou inválido, criar edges básicas (4-conectividade)
        if edge_index is None:
            print("   ⚠️ Criando edges sintéticas...")
            grid_size = int(np.sqrt(n_terrain))
            edge_list = []
            for i in range(min(n_terrain, grid_size * grid_size)):
                xi, yi = i % grid_size, i // grid_size
                if xi < grid_size - 1 and i + 1 < n_terrain:
                    edge_list.append([i, i + 1])
                    edge_list.append([i + 1, i])
                if yi < grid_size - 1 and i + grid_size < n_terrain:
                    edge_list.append([i, i + grid_size])
                    edge_list.append([i + grid_size, i])
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T
            print(f"   Edges criadas: {edge_index.shape[1]}")
        
        # Calcular edge_attr (distância entre nós conectados)
        src, dst = edge_index
        pos = data['terrain'].pos
        
        if pos.shape[1] >= 2:
            dx = pos[dst, 0] - pos[src, 0]
            dy = pos[dst, 1] - pos[src, 1]
            distances = torch.sqrt(dx**2 + dy**2)
        else:
            distances = torch.ones(edge_index.shape[1], device=device)
        
        edge_attr = distances.unsqueeze(1)
        
        # Extrair canopy dos features (assume que é a última coluna ou índice específico)
        # Por padrão, usar zeros se não disponível
        terrain_x = data['terrain'].x
        if terrain_x.shape[1] > 10:
            # Assume canopy está em posição fixa (ex: coluna 9 ou 10)
            canopy = terrain_x[:, 9].clamp(min=0)
        else:
            canopy = torch.zeros(n_terrain, device=device)
        
        # Encontrar nós de terreno mais próximos das antenas
        if 'antenna' in data.node_types:
            antenna_pos = data['antenna'].pos
            n_antennas = antenna_pos.shape[0]
            
            # Para cada antena, encontrar o nó de terreno mais próximo
            antenna_terrain_nodes = []
            for i in range(n_antennas):
                ant_x, ant_y = antenna_pos[i, 0], antenna_pos[i, 1]
                dist_to_terrain = torch.sqrt(
                    (pos[:, 0] - ant_x)**2 + (pos[:, 1] - ant_y)**2
                )
                nearest = dist_to_terrain.argmin()
                antenna_terrain_nodes.append(nearest.item())
            
            # Inicializar RSSI nos nós de antena com EIRP
            for node_idx in antenna_terrain_nodes:
                rssi[node_idx] = self.eirp_dbm
                hops_to_reach[node_idx] = 0
        else:
            # Sem antenas definidas, usar nó central
            center_node = n_terrain // 2
            rssi[center_node] = self.eirp_dbm
            hops_to_reach[center_node] = 0
        
        # Armazenar histórico por hop (se solicitado)
        rssi_history = [rssi.clone()] if return_per_hop else None
        
        # Propagação iterativa
        for hop in range(self.max_hops):
            rssi_old = rssi.clone()
            
            # Propagar sinal (Physics Layer)
            rssi = self.rf_layer(
                rssi=rssi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                canopy=canopy
            )
            # Atualizar hops para nós recém-atingidos
            newly_reached = (rssi > self.rssi_threshold) & (hops_to_reach < 0)
            hops_to_reach[newly_reached] = hop + 1
            
            if return_per_hop:
                rssi_history.append(rssi.clone())
            
            # Verificar convergência
            delta = (rssi - rssi_old).abs().max()
            if delta < 0.1:  # Convergiu
                break
        
        # Calcular path loss
        path_loss = self.eirp_dbm - rssi
        path_loss = torch.where(
            rssi > self.rssi_threshold,
            path_loss,
            torch.full_like(path_loss, float('inf'))
        )
        
        # Máscara de cobertura
        coverage_mask = rssi > self.rssi_threshold
        
        result = {
            'rssi': rssi,
            'path_loss': path_loss,
            'coverage_mask': coverage_mask,
            'hops_to_reach': hops_to_reach,
            'n_covered': coverage_mask.sum().item(),
            'coverage_pct': coverage_mask.float().mean().item() * 100
        }
        
        if return_per_hop:
            result['rssi_history'] = torch.stack(rssi_history)
        
        return result
    
    def set_antenna_params(
        self,
        tx_power_dbm: Optional[float] = None,
        tx_gain_dbi: Optional[float] = None,
        frequency_mhz: Optional[float] = None
    ):
        """Atualiza parâmetros da antena."""
        if tx_power_dbm is not None:
            self.tx_power_dbm = tx_power_dbm
        if tx_gain_dbi is not None:
            self.tx_gain_dbi = tx_gain_dbi
        if frequency_mhz is not None:
            self.frequency_mhz = frequency_mhz
            self.message_layer.frequency_mhz = frequency_mhz
        
        self.eirp_dbm = self.tx_power_dbm + self.tx_gain_dbi


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTE: PhysicsRFPropagator")
    print("=" * 70)
    
    # Criar grafo sintético
    print("\n## 1. Criando Grafo Sintético")
    print("-" * 50)
    
    # Grid 50x50 = 2500 nós
    grid_size = 50
    n_nodes = grid_size * grid_size
    
    # Posições em grid
    x_coords = torch.arange(grid_size).repeat(grid_size)
    y_coords = torch.arange(grid_size).repeat_interleave(grid_size)
    pos = torch.stack([x_coords, y_coords], dim=1).float()
    
    # Features (elevação, slope, canopy, etc.)
    torch.manual_seed(42)
    n_features = 17
    terrain_x = torch.randn(n_nodes, n_features)
    terrain_x[:, 0] = 100 + 20 * torch.sin(x_coords / 10) * torch.cos(y_coords / 10)  # Elevação
    terrain_x[:, 9] = torch.abs(torch.randn(n_nodes)) * 5  # Canopy (0-15m)
    
    # Criar arestas (4-conectividade)
    edge_list = []
    for i in range(n_nodes):
        x, y = i % grid_size, i // grid_size
        # Direita
        if x < grid_size - 1:
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])
        # Baixo
        if y < grid_size - 1:
            edge_list.append([i, i + grid_size])
            edge_list.append([i + grid_size, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    
    # Antenas (2 antenas)
    antenna_pos = torch.tensor([[10.0, 10.0], [40.0, 40.0]])
    antenna_x = torch.randn(2, 10)
    
    # Criar HeteroData
    data = HeteroData()
    data['terrain'].x = terrain_x
    data['terrain'].pos = pos
    data['antenna'].x = antenna_x
    data['antenna'].pos = antenna_pos
    data['terrain', 'connects_to', 'terrain'].edge_index = edge_index
    
    print(f"   Terrain nodes: {n_nodes}")
    print(f"   Edges: {edge_index.shape[1]}")
    print(f"   Antennas: {antenna_pos.shape[0]}")
    
    print("\n## 2. Testando PhysicsRFMessage")
    print("-" * 50)
    
    message_layer = PhysicsRFMessage(frequency_mhz=1800, cell_size_m=30)
    
    # RSSI inicial (-inf exceto nos nós de antena)
    rssi_init = torch.full((n_nodes,), float('-inf'))
    rssi_init[10 * 50 + 10] = 58  # EIRP = 43 + 15 = 58 dBm
    
    canopy = terrain_x[:, 9].clamp(min=0)
    distances = torch.ones(edge_index.shape[1])
    edge_attr = distances.unsqueeze(1)
    
    rssi_out = message_layer(rssi_init, edge_index, edge_attr, canopy)
    
    reached = (rssi_out > -120).sum()
    print(f"   Após 1 hop: {reached} nós atingidos")
    
    print("\n## 3. Testando PhysicsRFPropagator")
    print("-" * 50)
    
    propagator = PhysicsRFPropagator(
        frequency_mhz=1800,
        tx_power_dbm=43,
        tx_gain_dbi=15,
        max_hops=15,
        cell_size_m=30
    )
    
    result = propagator(data, return_per_hop=True)
    
    print(f"   RSSI range: {result['rssi'][result['coverage_mask']].min():.1f} to {result['rssi'].max():.1f} dBm")
    print(f"   Nodes covered: {result['n_covered']} ({result['coverage_pct']:.1f}%)")
    print(f"   Max hops used: {result['hops_to_reach'].max().item()}")
    
    print("\n## 4. Análise de Cobertura")
    print("-" * 50)
    
    # Estatísticas por distância da antena
    ant1_pos = antenna_pos[0]
    dist_to_ant1 = torch.sqrt((pos[:, 0] - ant1_pos[0])**2 + (pos[:, 1] - ant1_pos[1])**2)
    
    for max_dist in [5, 10, 20, 30]:
        in_range = dist_to_ant1 <= max_dist
        covered_in_range = (result['coverage_mask'] & in_range).sum()
        total_in_range = in_range.sum()
        pct = covered_in_range / total_in_range * 100 if total_in_range > 0 else 0
        avg_rssi = result['rssi'][in_range & result['coverage_mask']].mean() if covered_in_range > 0 else float('nan')
        print(f"   Distância ≤{max_dist}: {covered_in_range}/{total_in_range} cobertos ({pct:.1f}%), RSSI avg={avg_rssi:.1f} dBm")
    
    print("\n" + "=" * 70)
    print("✅ TESTES CONCLUÍDOS")
    print("=" * 70)
