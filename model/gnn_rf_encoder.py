

"""Heterogeneous GNN encoder for RF propagation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


class TerrainEncoder(nn.Module):
    """Encoder MLP para features de terreno (17 dim -> hidden_dim)."""

    def __init__(self,
                 in_channels: int = 17,
                 hidden_channels: int = 256,
                 out_channels: int = 512):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AntennaEncoder(nn.Module):
    """Encoder MLP para features de antena (10 dim -> hidden_dim)."""

    def __init__(self,
                 in_channels: int = 10,
                 hidden_channels: int = 128,
                 out_channels: int = 512):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GNNRFEncoder(nn.Module):
    """
    Heterogeneous GNN encoder for RF propagation.

    Architecture:
        hidden_dim: 512
        heads: 4
        num_layers: 4
        edge_dim: 2
    """

    def __init__(self,
                 terrain_dim: int = 17,
                 antenna_dim: int = 10,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 heads: int = 4,
                 edge_dim: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input encoders
        self.terrain_encoder = TerrainEncoder(
            in_channels=terrain_dim,
            hidden_channels=256,
            out_channels=hidden_dim
        )

        self.antenna_encoder = AntennaEncoder(
            in_channels=antenna_dim,
            hidden_channels=128,
            out_channels=hidden_dim
        )


        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            conv = HeteroConv({

                ('antenna', 'propagates_to', 'terrain'): GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=False
                ),

                ('terrain', 'connects_to', 'terrain'): GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True
                ),

                ('terrain', 'in_range_of', 'antenna'): SAGEConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggr='mean'
                )
            }, aggr='sum')

            self.convs.append(conv)


            self.layer_norms.append(nn.ModuleDict({
                'terrain': nn.LayerNorm(hidden_dim),
                'antenna': nn.LayerNorm(hidden_dim)
            }))

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Encoder forward pass.

        Args:
            data: HeteroData with 'terrain' and 'antenna' node types.

        Returns:
            Dict of embeddings per node type.
        """

        x_dict = {}
        device = data['terrain'].x.device if hasattr(data['terrain'], 'x') else torch.device('cpu')


        if hasattr(data['terrain'], 'x') and data['terrain'].num_nodes > 0:
            x_dict['terrain'] = self.terrain_encoder(data['terrain'].x)
        else:
             # Should practically not happen as we seed with terrain
            x_dict['terrain'] = torch.zeros((0, self.hidden_dim), device=device)


        if hasattr(data['antenna'], 'x') and data['antenna'].num_nodes > 0:
            x_dict['antenna'] = self.antenna_encoder(data['antenna'].x)
        else:


            x_dict['antenna'] = torch.zeros((0, self.hidden_dim), device=device)


        edge_index_dict = {}
        edge_attr_dict = {}

        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index
            if hasattr(data[edge_type], 'edge_attr'):
                edge_attr_dict[edge_type] = data[edge_type].edge_attr


        for i, conv in enumerate(self.convs):

            x_res = {k: v.clone() for k, v in x_dict.items() if v is not None}


            # HeteroConv expects x_dict to have keys for every node type involved


            try:

                out_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            except Exception as e:


                out_dict = conv(x_dict, edge_index_dict)


            for node_type, out_feat in out_dict.items():
                if out_feat is None:
                    continue


                out_feat = F.leaky_relu(out_feat, 0.2)


                out_feat = F.dropout(
                    out_feat,
                    p=self.dropout,
                    training=self.training
                )


                if node_type in x_res:
                    # Check shape match (sometimes GNN changes shape or target nodes differ)
                    if x_res[node_type].shape == out_feat.shape:
                        out_feat = out_feat + x_res[node_type]


                out_feat = self.layer_norms[i][node_type](out_feat)


                x_dict[node_type] = out_feat

        return x_dict

    def get_terrain_embeddings(self, data: HeteroData) -> torch.Tensor:
        """Retorna apenas embeddings de terreno."""
        x_dict = self.forward(data)
        return x_dict['terrain']


def validate_encoder(encoder: GNNRFEncoder,
                    hetero_data: HeteroData,
                    verbose: bool = True) -> bool:
    """
    Validates the GNN-RF encoder.

    Args:
        encoder: model to validate.
        hetero_data: test data.

    Returns:
        bool: True if valid.
    """
    errors = []

    try:
        encoder.eval()
        with torch.no_grad():
            x_dict = encoder(hetero_data)


        if 'terrain' not in x_dict:
            errors.append("Missing 'terrain' in output")
        elif x_dict['terrain'].shape[1] != encoder.hidden_dim:
            errors.append(f"terrain hidden_dim mismatch: {x_dict['terrain'].shape[1]} != {encoder.hidden_dim}")

        if 'antenna' not in x_dict:
            errors.append("Missing 'antenna' in output")
        elif x_dict['antenna'].shape[1] != encoder.hidden_dim:
            errors.append(f"antenna hidden_dim mismatch: {x_dict['antenna'].shape[1]} != {encoder.hidden_dim}")


        for key, val in x_dict.items():
            if val is not None and torch.isnan(val).any():
                errors.append(f"{key} embeddings contain NaN")

    except Exception as e:
        errors.append(f"Forward pass failed: {str(e)}")

    if errors:
        for e in errors:
            print(f"❌ VALIDATE Error: {e}")
        return False

    if verbose:
        print(f"✅ validate_encoder PASSED")
        print(f"   Terrain embeddings: {x_dict['terrain'].shape}")
        print(f"   Antenna embeddings: {x_dict['antenna'].shape}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: GNNRFEncoder")
    print("=" * 60)


    from torch_geometric.data import HeteroData

    n_terrain = 500
    n_antenna = 3
    n_prop_edges = 800
    n_t2t_edges = 2000

    data = HeteroData()
    data['terrain'].x = torch.randn(n_terrain, 17)
    data['antenna'].x = torch.randn(n_antenna, 10)

    data['antenna', 'propagates_to', 'terrain'].edge_index = torch.stack([
        torch.randint(0, n_antenna, (n_prop_edges,)),
        torch.randint(0, n_terrain, (n_prop_edges,))
    ])
    data['antenna', 'propagates_to', 'terrain'].edge_attr = torch.randn(n_prop_edges, 2)

    data['terrain', 'connects_to', 'terrain'].edge_index = torch.stack([
        torch.randint(0, n_terrain, (n_t2t_edges,)),
        torch.randint(0, n_terrain, (n_t2t_edges,))
    ])
    data['terrain', 'connects_to', 'terrain'].edge_attr = torch.randn(n_t2t_edges, 2)

    data['terrain', 'in_range_of', 'antenna'].edge_index = torch.stack([
        torch.randint(0, n_terrain, (n_prop_edges,)),
        torch.randint(0, n_antenna, (n_prop_edges,))
    ])


    print("\n📊 Criando encoder...")
    encoder = GNNRFEncoder(
        terrain_dim=17,
        antenna_dim=10,
        hidden_dim=512,
        num_layers=4,
        heads=4
    )

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"   Total parameters: {total_params:,}")


    is_valid = validate_encoder(encoder, data)

    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if is_valid else "❌ TESTE FALHOU")
    print("=" * 60)
