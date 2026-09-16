

"""
Physics-based RF propagator using message passing.

Propagates the RF signal through the graph using physical models:
    - Free Space Path Loss.
    - Vegetation (ITU-R P.833).
    - Diffraction (Knife-Edge).
    - LOS/NLOS detection.

The signal is propagated iteratively from antenna nodes to terrain
nodes, computing RSSI at each node reached.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


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


DEFAULT_FREQUENCY_MHZ = 1800.0
DEFAULT_TX_POWER_DBM = 43.0
DEFAULT_TX_GAIN_DBI = 15.0
DEFAULT_RX_GAIN_DBI = 0.0
DEFAULT_RSSI_THRESHOLD = -120.0
DEFAULT_CELL_SIZE_M = 30.0


class PhysicsRFMessage(nn.Module):
    """
    RF propagation layer with physics-based attenuation.

    Propagates the RF signal from antenna nodes to terrain nodes, applying
    physical attenuation based on:
    - distance (FSPL);
    - vegetation (canopy height);
    - diffraction (obstruction height).

    Uses scatter_max to aggregate messages.

    Args:
        frequency_mhz: operating frequency.
        cell_size_m: cell size in meters.
        rssi_threshold: minimum RSSI to continue propagation.
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
        Propagates the RF signal across the graph edges.

        Args:
            rssi: [N] current RSSI at each node (dBm).
            edge_index: [2, E] edge indices.
            edge_attr: [E, D] edge attributes (distance, etc.).
            canopy: [N] canopy height at each node (optional).

        Returns:
            rssi_updated: [N] RSSI after propagation.
        """
        src, dst = edge_index
        n_nodes = rssi.shape[0]
        device = rssi.device

        if canopy is None:
            canopy = torch.zeros(n_nodes, device=device)


        if edge_attr.dim() == 1:
            distance_cells = edge_attr
        else:
            distance_cells = edge_attr[:, 0]

        distance_m = distance_cells * self.cell_size_m
        distance_m = torch.clamp(distance_m, min=self.cell_size_m)


        rssi_src = rssi[src]


        canopy_src = canopy[src]
        canopy_dst = canopy[dst]
        avg_canopy = (canopy_src + canopy_dst) / 2
        veg_path_m = distance_m * 0.3
        veg_loss = vegetation_loss_itu_p833(veg_path_m, self.frequency_mhz, avg_canopy)


        if edge_attr.dim() > 1 and edge_attr.shape[1] > 1:
            elev_diff = edge_attr[:, 1]
            obstruction = torch.clamp(elev_diff, min=0)
            d1 = distance_m / 2
            d2 = distance_m / 2
            diff_loss = knife_edge_diffraction_loss(obstruction, d1, d2, self.frequency_mhz)
        else:
            diff_loss = torch.zeros_like(veg_loss)


        # K = 20*log10(f) - 27.55
        eirp = self.tx_power_dbm + self.tx_gain_dbi
        k_factor = 20 * torch.log10(torch.tensor(self.frequency_mhz, device=device)) - 27.55

        # Avoid log of infinity: clamp RSSI


        rssi_clamped = torch.clamp(rssi_src, max=eirp - 1.0) # Avoid d=0

        val = (eirp - rssi_clamped - k_factor) / 20.0
        d_virt = torch.pow(10.0, val)


        d_virt = torch.clamp(d_virt, min=1.0)


        d_new = d_virt + distance_m


        # Loss_geo = 20 * log10(d_new / d_virt)
        geo_loss_db = 20 * torch.log10(d_new / d_virt)


        linear_losses = veg_loss + diff_loss


        total_loss_hop = geo_loss_db + linear_losses


        rssi_received = rssi_src - total_loss_hop


        # Apply threshold; signals below it do not propagate
        valid_mask = rssi_received >= self.rssi_threshold


        rssi_new = torch.full((n_nodes,), float('-inf'), device=device)

        if valid_mask.any():

            valid_dst = dst[valid_mask]
            valid_rssi = rssi_received[valid_mask]


            rssi_new.scatter_reduce_(0, valid_dst, valid_rssi, reduce='amax', include_self=False)


        rssi_updated = torch.maximum(rssi, rssi_new)

        return rssi_updated


class PhysicsRFPropagator(nn.Module):
    """
    Full physics-based message-passing RF propagator.

    Propagates the signal iteratively from the antennas to the whole
    terrain, using physical models to compute attenuation.

    Args:
        frequency_mhz: operating frequency.
        tx_power_dbm: transmit power.
        tx_gain_dbi: TX antenna gain.
        max_hops: maximum number of propagation hops.
        cell_size_m: cell size in meters.
        rssi_threshold: minimum RSSI threshold.
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


        self.eirp_dbm = tx_power_dbm + tx_gain_dbi


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
        Propagates the RF signal from the antennas to the whole terrain.

        Args:
            data: HeteroData with:
                - data['terrain'].x: terrain features [N, F].
                - data['terrain'].pos: positions [N, 2].
                - data['antenna'].x: antenna features [M, F].
                - data['antenna'].pos: antenna positions [M, 2].
                - data[('terrain', 'connects_to', 'terrain')].edge_index: edges.
            antenna_indices: indices of active antennas (optional).
            return_per_hop: if True, also returns RSSI at each hop.

        Returns:
            Dict with:
                - 'rssi': [N] final RSSI at each terrain node.
                - 'path_loss': [N] total path loss.
                - 'coverage_mask': [N] coverage mask (> threshold).
                - 'hops_to_reach': [N] number of hops to reach each node.
        """
        device = data['terrain'].x.device
        n_terrain = data['terrain'].x.shape[0]


        rssi = torch.full((n_terrain,), float('-inf'), device=device)
        hops_to_reach = torch.full((n_terrain,), -1, dtype=torch.int32, device=device)


        edge_key = ('terrain', 'connects_to', 'terrain')
        edge_index = None


        possible_keys = [
            ('terrain', 'connects_to', 'terrain'),
            ('terrain', 'to', 'terrain'),
        ]

        for key in possible_keys:
            if key in data.edge_types:
                candidate = data[key].edge_index

                if candidate.max() < n_terrain:
                    edge_index = candidate
                    break


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


        src, dst = edge_index
        pos = data['terrain'].pos

        if pos.shape[1] >= 2:
            dx = pos[dst, 0] - pos[src, 0]
            dy = pos[dst, 1] - pos[src, 1]
            distances = torch.sqrt(dx**2 + dy**2)
        else:
            distances = torch.ones(edge_index.shape[1], device=device)

        edge_attr = distances.unsqueeze(1)

        # Extracts canopy height from the features (assumes it is the last column or a specific index)

        terrain_x = data['terrain'].x
        if terrain_x.shape[1] > 10:
            # Assumes canopy height is at a fixed position (e.g. column 9 or 10)
            canopy = terrain_x[:, 9].clamp(min=0)
        else:
            canopy = torch.zeros(n_terrain, device=device)


        if 'antenna' in data.node_types:
            antenna_pos = data['antenna'].pos
            n_antennas = antenna_pos.shape[0]


            antenna_terrain_nodes = []
            for i in range(n_antennas):
                ant_x, ant_y = antenna_pos[i, 0], antenna_pos[i, 1]
                dist_to_terrain = torch.sqrt(
                    (pos[:, 0] - ant_x)**2 + (pos[:, 1] - ant_y)**2
                )
                nearest = dist_to_terrain.argmin()
                antenna_terrain_nodes.append(nearest.item())


            for node_idx in antenna_terrain_nodes:
                rssi[node_idx] = self.eirp_dbm
                hops_to_reach[node_idx] = 0
        else:

            center_node = n_terrain // 2
            rssi[center_node] = self.eirp_dbm
            hops_to_reach[center_node] = 0


        rssi_history = [rssi.clone()] if return_per_hop else None


        for hop in range(self.max_hops):
            rssi_old = rssi.clone()


            rssi = self.rf_layer(
                rssi=rssi,
                edge_index=edge_index,
                edge_attr=edge_attr,
                canopy=canopy
            )

            newly_reached = (rssi > self.rssi_threshold) & (hops_to_reach < 0)
            hops_to_reach[newly_reached] = hop + 1

            if return_per_hop:
                rssi_history.append(rssi.clone())


            delta = (rssi - rssi_old).abs().max()
            if delta < 0.1:
                break


        path_loss = self.eirp_dbm - rssi
        path_loss = torch.where(
            rssi > self.rssi_threshold,
            path_loss,
            torch.full_like(path_loss, float('inf'))
        )


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
        """Updates the antenna parameters."""
        if tx_power_dbm is not None:
            self.tx_power_dbm = tx_power_dbm
        if tx_gain_dbi is not None:
            self.tx_gain_dbi = tx_gain_dbi
        if frequency_mhz is not None:
            self.frequency_mhz = frequency_mhz
            self.message_layer.frequency_mhz = frequency_mhz

        self.eirp_dbm = self.tx_power_dbm + self.tx_gain_dbi


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTE: PhysicsRFPropagator")
    print("=" * 70)


    print("\n## 1. Criando Grafo Sintético")
    print("-" * 50)

    # 50x50 grid = 2500 nodes
    grid_size = 50
    n_nodes = grid_size * grid_size


    x_coords = torch.arange(grid_size).repeat(grid_size)
    y_coords = torch.arange(grid_size).repeat_interleave(grid_size)
    pos = torch.stack([x_coords, y_coords], dim=1).float()


    torch.manual_seed(42)
    n_features = 17
    terrain_x = torch.randn(n_nodes, n_features)
    terrain_x[:, 0] = 100 + 20 * torch.sin(x_coords / 10) * torch.cos(y_coords / 10)
    terrain_x[:, 9] = torch.abs(torch.randn(n_nodes)) * 5


    edge_list = []
    for i in range(n_nodes):
        x, y = i % grid_size, i // grid_size

        if x < grid_size - 1:
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])

        if y < grid_size - 1:
            edge_list.append([i, i + grid_size])
            edge_list.append([i + grid_size, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).T


    antenna_pos = torch.tensor([[10.0, 10.0], [40.0, 40.0]])
    antenna_x = torch.randn(2, 10)


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
