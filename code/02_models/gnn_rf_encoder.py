# GNN-RF Propagation V2
# 02_models/gnn_rf_encoder.py

"""
Encoder GNN heterogêneo para propagação RF.
Compatível com arquitetura TerrainGNN v21 (hidden_dim=512, heads=4).

Processa:
    - Nós de terreno: [N, 17] -> [N, hidden_dim]
    - Nós de antena: [M, 10] -> [M, hidden_dim]
    - Arestas heterogêneas com edge_attr [E, 2]
"""

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
    Encoder GNN heterogêneo para propagação RF.
    
    Arquitetura compatível com TerrainGNN v21:
        - hidden_dim: 512
        - heads: 4
        - num_layers: 4
        - edge_dim: 2
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
        
        # Encoders de entrada
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
        
        # Camadas GNN heterogêneas
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            conv = HeteroConv({
                # Propagação antena -> terreno
                ('antenna', 'propagates_to', 'terrain'): GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=False
                ),
                # Conexão terreno -> terreno
                ('terrain', 'connects_to', 'terrain'): GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True
                ),
                # Reverso terreno -> antena (para atualizar antenas)
                ('terrain', 'in_range_of', 'antenna'): SAGEConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggr='mean'
                )
            }, aggr='sum')
            
            self.convs.append(conv)
            
            # Layer norms para cada tipo de nó
            self.layer_norms.append(nn.ModuleDict({
                'terrain': nn.LayerNorm(hidden_dim),
                'antenna': nn.LayerNorm(hidden_dim)
            }))
    
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward pass do encoder.
        
        Args:
            data: HeteroData com 'terrain' e 'antenna'
        
        Returns:
            Dict com embeddings por tipo de nó
        """
        # Encode inputs
        x_dict = {}
        device = data['terrain'].x.device if hasattr(data['terrain'], 'x') else torch.device('cpu')
        
        # Terrain
        if hasattr(data['terrain'], 'x') and data['terrain'].num_nodes > 0:
            x_dict['terrain'] = self.terrain_encoder(data['terrain'].x)
        else:
             # Should practically not happen as we seed with terrain
            x_dict['terrain'] = torch.zeros((0, self.hidden_dim), device=device)
            
        # Antenna
        if hasattr(data['antenna'], 'x') and data['antenna'].num_nodes > 0:
            x_dict['antenna'] = self.antenna_encoder(data['antenna'].x)
        else:
            # Handle empty antenna batch (e.g. isolated terrain nodes)
            # Create 0-sized tensor with correct dim to keep GNN happy
            x_dict['antenna'] = torch.zeros((0, self.hidden_dim), device=device)

        # Preparar edge_index_dict
        edge_index_dict = {}
        edge_attr_dict = {}
        
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index
            if hasattr(data[edge_type], 'edge_attr'):
                edge_attr_dict[edge_type] = data[edge_type].edge_attr
        
        # Aplicar camadas GNN
        for i, conv in enumerate(self.convs):
            # Salvar para residual
            x_res = {k: v.clone() for k, v in x_dict.items() if v is not None}
            
            # Convolução heterogênea
            # HeteroConv espera que x_dict tenha chaves para todos tipos de nos envolvidos
            # Se algum estiver faltando (batching pode causar isso), precisamos lidar
            
            try:
                # Checar se temos inputs suficientes
                out_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            except Exception as e:
                # Fallback: tentar sem edge_attr
                # print(f"⚠️ Conv Error (Layer {i}): {e}. Trying without edge attributes.")
                out_dict = conv(x_dict, edge_index_dict)
            
            # Atualizar x_dict com resultados
            # Activation + Dropout + Residual + LayerNorm
            for node_type, out_feat in out_dict.items():
                if out_feat is None:
                    continue
                    
                # Apply activation
                out_feat = F.leaky_relu(out_feat, 0.2)
                
                # Dropout
                out_feat = F.dropout(
                    out_feat, 
                    p=self.dropout, 
                    training=self.training
                )
                
                # Residual
                if node_type in x_res:
                    # Check shape match (sometimes GNN changes shape or target nodes differ)
                    if x_res[node_type].shape == out_feat.shape:
                        out_feat = out_feat + x_res[node_type]
                
                # Layer Norm
                out_feat = self.layer_norms[i][node_type](out_feat)
                
                # Update dict for next layer
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
    Valida encoder GNN-RF.
    
    Args:
        encoder: Modelo a validar
        hetero_data: Dados de teste
    
    Returns:
        bool: True se válido
    """
    errors = []
    
    try:
        encoder.eval()
        with torch.no_grad():
            x_dict = encoder(hetero_data)
        
        # Verificar outputs
        if 'terrain' not in x_dict:
            errors.append("Missing 'terrain' in output")
        elif x_dict['terrain'].shape[1] != encoder.hidden_dim:
            errors.append(f"terrain hidden_dim mismatch: {x_dict['terrain'].shape[1]} != {encoder.hidden_dim}")
        
        if 'antenna' not in x_dict:
            errors.append("Missing 'antenna' in output")
        elif x_dict['antenna'].shape[1] != encoder.hidden_dim:
            errors.append(f"antenna hidden_dim mismatch: {x_dict['antenna'].shape[1]} != {encoder.hidden_dim}")
        
        # Verificar NaN
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
    
    # Criar dados sintéticos
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
    
    # Criar encoder
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
    
    # Validar
    is_valid = validate_encoder(encoder, data)
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if is_valid else "❌ TESTE FALHOU")
    print("=" * 60)
