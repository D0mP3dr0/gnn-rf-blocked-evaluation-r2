# GNN-RF Propagation V2
# 02_models/__init__.py

"""
Módulo de modelos para GNN-RF Propagation.

Componentes:
    - GNNRFEncoder: Encoder heterogêneo (terrain + antenna)
    - RFDecoder: Decoder para predições RF
    - PhysicsConstrainedDecoder: Decoder com constraints físicos
    - PhysicsRFLoss: Loss physics-informed
    - CurriculumRFLoss: Loss com curriculum learning
    - GNNRFModel: Modelo completo
"""

try:
    from .gnn_rf_encoder import GNNRFEncoder, validate_encoder
    from .rf_decoder import RFDecoder, PhysicsConstrainedDecoder, validate_decoder
    from .physics_loss import PhysicsRFLoss, CurriculumRFLoss, validate_physics_loss
    from .gnn_rf_model import GNNRFModel, GNNRFLightningModule, validate_full_model
except ImportError:
    from gnn_rf_encoder import GNNRFEncoder, validate_encoder
    from rf_decoder import RFDecoder, PhysicsConstrainedDecoder, validate_decoder
    from physics_loss import PhysicsRFLoss, CurriculumRFLoss, validate_physics_loss
    from gnn_rf_model import GNNRFModel, GNNRFLightningModule, validate_full_model


__all__ = [
    # Encoder
    'GNNRFEncoder',
    'validate_encoder',
    
    # Decoder
    'RFDecoder',
    'PhysicsConstrainedDecoder',
    'validate_decoder',
    
    # Loss
    'PhysicsRFLoss',
    'CurriculumRFLoss',
    'validate_physics_loss',
    
    # Model
    'GNNRFModel',
    'GNNRFLightningModule',
    'validate_full_model'
]
