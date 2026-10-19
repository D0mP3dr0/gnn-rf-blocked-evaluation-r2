# GNN-RF Propagation V2
# 02_models/gnn_rf_model.py

"""
Modelo GNN-RF completo (Encoder + Decoder).
Integra todos os componentes para predição de propagação RF.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

# Imports condicionais
try:
    from .gnn_rf_encoder import GNNRFEncoder
    from .rf_decoder import RFDecoder, PhysicsConstrainedDecoder
    from .physics_loss import PhysicsRFLoss, CurriculumRFLoss
except ImportError:
    from gnn_rf_encoder import GNNRFEncoder
    from rf_decoder import RFDecoder, PhysicsConstrainedDecoder
    from physics_loss import PhysicsRFLoss, CurriculumRFLoss


class GNNRFModel(nn.Module):
    """
    Modelo completo GNN-RF para predição de propagação.
    
    Arquitetura:
        Input: HeteroData (terrain: 17 dim, antenna: 10 dim)
        Encoder: GNNRFEncoder (hidden_dim: 512)
        Decoder: RFDecoder (output: 5 dim)
        Output: Predições de path_loss, RSSI, coverage
    """
    
    def __init__(self,
                 terrain_dim: int = 17,
                 antenna_dim: int = 10,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 heads: int = 4,
                 edge_dim: int = 2,
                 output_dim: int = 5,
                 dropout: float = 0.1,
                 use_physics_constraints: bool = True):
        super().__init__()
        
        self.terrain_dim = terrain_dim
        self.antenna_dim = antenna_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encoder
        self.encoder = GNNRFEncoder(
            terrain_dim=terrain_dim,
            antenna_dim=antenna_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout
        )
        
        # Decoder
        if use_physics_constraints:
            self.decoder = PhysicsConstrainedDecoder(
                in_channels=hidden_dim,
                hidden_channels=256,
                out_channels=output_dim,
                dropout=dropout
            )
        else:
            self.decoder = RFDecoder(
                in_channels=hidden_dim,
                hidden_channels=256,
                out_channels=output_dim,
                dropout=dropout
            )
    
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo.
        
        Args:
            data: HeteroData com terrain e antenna
        
        Returns:
            Dict com:
                - terrain_embeddings: [N, hidden_dim]
                - antenna_embeddings: [M, hidden_dim]
                - predictions: [N, 5]
                - decoded: Dict com predições nomeadas
        """
        # Encode
        embeddings = self.encoder(data)
        
        # Decode (apenas terreno)
        terrain_embeddings = embeddings['terrain']
        predictions = self.decoder(terrain_embeddings)
        
        # Decodificar nomes
        decoded = self.decoder.decode_predictions(predictions)
        
        return {
            'terrain_embeddings': terrain_embeddings,
            'antenna_embeddings': embeddings['antenna'],
            'predictions': predictions,
            'decoded': decoded
        }
    
    def predict(self, data: HeteroData) -> torch.Tensor:
        """Retorna apenas predições [N, 5]."""
        output = self.forward(data)
        return output['predictions']
    
    def get_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Retorna embeddings de terrain e antenna."""
        return self.encoder(data)
    
    def count_parameters(self) -> Dict[str, int]:
        """Conta parâmetros do modelo."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = encoder_params + decoder_params
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total
        }


class GNNRFLightningModule:
    """
    Wrapper para PyTorch Lightning (apenas interface, sem dependência).
    
    Define interface para treinamento:
        - training_step
        - validation_step
        - configure_optimizers
    """
    
    def __init__(self,
                 model: GNNRFModel,
                 loss_fn: PhysicsRFLoss,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def training_step(self, batch: HeteroData) -> Tuple[torch.Tensor, Dict]:
        """
        Um passo de treinamento.
        
        Args:
            batch: HeteroData com dados e targets
        
        Returns:
            loss: Scalar
            metrics: Dict com métricas
        """
        # Forward
        output = self.model(batch)
        predictions = output['predictions']
        
        # Targets
        targets = batch['terrain'].rf_targets
        
        # Canopy (se disponível)
        canopy = None
        if hasattr(batch['terrain'], 'y'):
            canopy = batch['terrain'].y[:, config.terrain.canopy_index]
        
        # Loss
        loss, loss_dict = self.loss_fn(predictions, targets, canopy_height=canopy)
        
        # Métricas
        with torch.no_grad():
            mae = torch.abs(predictions - targets).mean(dim=0)
            metrics = {
                'train_loss': loss.item(),
                'mae_path_loss': mae[0].item(),
                'mae_rssi': mae[3].item()
            }
            metrics.update({k: v.item() for k, v in loss_dict.items()})
        
        return loss, metrics
    
    def validation_step(self, batch: HeteroData) -> Dict:
        """Passo de validação (sem gradientes)."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)
            predictions = output['predictions']
            targets = batch['terrain'].rf_targets
            
            # Métricas
            mae = torch.abs(predictions - targets).mean(dim=0)
            rmse = torch.sqrt(((predictions - targets) ** 2).mean(dim=0))
            
            metrics = {
                'val_mae_path_loss': mae[0].item(),
                'val_rmse_path_loss': rmse[0].item(),
                'val_mae_rssi': mae[3].item(),
                'val_rmse_rssi': rmse[3].item(),
                'val_mae_coverage': mae[4].item()
            }
        
        self.model.train()
        return metrics
    
    def configure_optimizers(self):
        """Configura otimizador e scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        return optimizer, scheduler


def validate_full_model(model: GNNRFModel,
                       hetero_data: HeteroData,
                       verbose: bool = True) -> bool:
    """
    Valida modelo completo.
    
    Args:
        model: GNNRFModel a validar
        hetero_data: Dados de teste
    
    Returns:
        bool: True se válido
    """
    errors = []
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(hetero_data)
        
        # Verificar outputs
        required_keys = ['terrain_embeddings', 'antenna_embeddings', 'predictions', 'decoded']
        for key in required_keys:
            if key not in output:
                errors.append(f"Missing output key: {key}")
        
        # Verificar shapes
        n_terrain = hetero_data['terrain'].x.shape[0]
        
        if output['terrain_embeddings'].shape != (n_terrain, model.hidden_dim):
            errors.append(f"terrain_embeddings shape mismatch")
        
        if output['predictions'].shape[1] != model.output_dim:
            errors.append(f"predictions dim mismatch: {output['predictions'].shape[1]} != {model.output_dim}")
        
        # Verificar decoded
        expected_decoded = ['path_loss_total', 'path_loss_vegetation', 'path_loss_terrain', 'rssi', 'coverage_prob']
        for key in expected_decoded:
            if key not in output['decoded']:
                errors.append(f"Missing decoded key: {key}")
        
        # Verificar NaN
        if torch.isnan(output['predictions']).any():
            errors.append("Predictions contain NaN")
        
    except Exception as e:
        errors.append(f"Forward pass failed: {str(e)}")
    
    if errors:
        for e in errors:
            print(f"❌ VALIDATE Error: {e}")
        return False
    
    if verbose:
        params = model.count_parameters()
        print(f"✅ validate_full_model PASSED")
        print(f"   Terrain embeddings: {output['terrain_embeddings'].shape}")
        print(f"   Predictions: {output['predictions'].shape}")
        print(f"   Parameters: {params['total']:,}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: GNNRFModel")
    print("=" * 60)
    
    # Criar dados sintéticos
    from torch_geometric.data import HeteroData
    
    n_terrain = 500
    n_antenna = 3
    n_prop_edges = 800
    n_t2t_edges = 2000
    
    data = HeteroData()
    data['terrain'].x = torch.randn(n_terrain, 17)
    data['terrain'].rf_targets = torch.randn(n_terrain, 5)
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
    
    # Criar modelo
    print("\n📊 Criando modelo completo...")
    model = GNNRFModel(
        terrain_dim=17,
        antenna_dim=10,
        hidden_dim=512,
        num_layers=4,
        heads=4,
        use_physics_constraints=True
    )
    
    params = model.count_parameters()
    print(f"   Encoder params: {params['encoder']:,}")
    print(f"   Decoder params: {params['decoder']:,}")
    print(f"   Total params: {params['total']:,}")
    
    # Validar
    is_valid = validate_full_model(model, data)
    
    # Testar Lightning wrapper
    print("\n📊 Testando Lightning wrapper...")
    loss_fn = CurriculumRFLoss()
    lightning_module = GNNRFLightningModule(model, loss_fn)
    
    loss, metrics = lightning_module.training_step(data)
    print(f"   Training loss: {loss.item():.4f}")
    print(f"   MAE path_loss: {metrics['mae_path_loss']:.4f}")
    
    val_metrics = lightning_module.validation_step(data)
    print(f"   Val MAE RSSI: {val_metrics['val_mae_rssi']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if is_valid else "❌ TESTE FALHOU")
    print("=" * 60)
