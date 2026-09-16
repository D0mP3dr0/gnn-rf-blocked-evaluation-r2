

"""Decoder for RF propagation predictions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


class RFDecoder(nn.Module):
    """
    MLP decoder for RF predictions.

    Converts terrain embeddings [N, hidden_dim] into RF predictions [N, 5].
    """

    def __init__(self,
                 in_channels: int = 512,
                 hidden_channels: int = 256,
                 out_channels: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        self.out_channels = out_channels


        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        # Dedicated heads per output type
        self.path_loss_head = nn.Linear(out_channels, 3)
        self.rssi_head = nn.Linear(out_channels, 1)
        self.coverage_head = nn.Linear(out_channels, 1)

    def forward(self,
                terrain_embeddings: torch.Tensor,
                use_separate_heads: bool = False
                ) -> torch.Tensor:
        """
        Decoder forward pass.

        Args:
            terrain_embeddings: [N, hidden_dim] terrain embeddings.
            use_separate_heads: whether to use separate output heads (more stable).

        Returns:
            predictions: [N, 5] RF predictions.
        """

        x = self.layers(terrain_embeddings)

        if use_separate_heads:

            path_loss = self.path_loss_head(x)
            rssi = self.rssi_head(x)
            coverage = torch.sigmoid(self.coverage_head(x))

            predictions = torch.cat([path_loss, rssi, coverage], dim=1)
        else:
            predictions = x

            predictions = torch.cat([
                predictions[:, :4],
                torch.sigmoid(predictions[:, 4:5])
            ], dim=1)

        return predictions

    def decode_predictions(self,
                          predictions: torch.Tensor
                          ) -> Dict[str, torch.Tensor]:
        """
        Decodes predictions into a named dictionary.

        Args:
            predictions: [N, 5] prediction tensor.

        Returns:
            Dict of named predictions.
        """
        return {
            'path_loss_total': predictions[:, 0],
            'path_loss_vegetation': predictions[:, 1],
            'path_loss_terrain': predictions[:, 2],
            'rssi': predictions[:, 3],
            'coverage_prob': predictions[:, 4]
        }


class PhysicsConstrainedDecoder(RFDecoder):
    """
    Decoder with physical constraints.

    Enforces:
        - path_loss_total >= 0
        - path_loss_vegetation >= 0
        - path_loss_terrain >= 0
        - rssi <= P_tx (transmit power)
        - coverage_prob in [0, 1]
    """

    def __init__(self,
                 in_channels: int = 512,
                 hidden_channels: int = 256,
                 min_path_loss: float = 0.0,
                 max_path_loss: float = 200.0,
                 min_rssi: float = -150.0,
                 max_rssi: float = 0.0,
                 **kwargs):
        super().__init__(in_channels, hidden_channels, **kwargs)

        self.min_path_loss = min_path_loss
        self.max_path_loss = max_path_loss
        self.min_rssi = min_rssi
        self.max_rssi = max_rssi

    def forward(self,
                terrain_embeddings: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Forward pass with physical constraints applied."""

        raw_predictions = super().forward(terrain_embeddings, use_separate_heads=False)


        constrained = torch.zeros_like(raw_predictions)

        # Path loss: softplus to ensure >= 0, then clamp
        constrained[:, 0] = torch.clamp(
            F.softplus(raw_predictions[:, 0]),
            self.min_path_loss,
            self.max_path_loss
        )
        constrained[:, 1] = torch.clamp(
            F.softplus(raw_predictions[:, 1]),
            0,
            50  # Vegetation loss capped at ~50 dB
        )
        constrained[:, 2] = torch.clamp(
            F.softplus(raw_predictions[:, 2]),
            0,
            30  # Terrain loss capped at ~30 dB
        )


        rssi_range = self.max_rssi - self.min_rssi
        constrained[:, 3] = self.min_rssi + rssi_range * torch.sigmoid(raw_predictions[:, 3])


        constrained[:, 4] = raw_predictions[:, 4]

        return constrained


def validate_decoder(decoder: RFDecoder,
                    embeddings: torch.Tensor,
                    verbose: bool = True) -> bool:
    """
    Validates the RF decoder.

    Args:
        decoder: model to validate.
        embeddings: [N, hidden_dim] test embeddings.

    Returns:
        bool: True if valid.
    """
    errors = []

    try:
        decoder.eval()
        with torch.no_grad():
            predictions = decoder(embeddings)

        # Check shape
        if predictions.shape[1] != 5:
            errors.append(f"Expected 5 outputs, got {predictions.shape[1]}")


        if torch.isnan(predictions).any():
            errors.append("Predictions contain NaN")


        if isinstance(decoder, PhysicsConstrainedDecoder):
            path_loss = predictions[:, 0]
            if (path_loss < 0).any():
                errors.append(f"path_loss < 0: min={path_loss.min():.2f}")

            rssi = predictions[:, 3]
            if rssi.min() < -200 or rssi.max() > 50:
                errors.append(f"RSSI out of range: [{rssi.min():.1f}, {rssi.max():.1f}]")

            coverage = predictions[:, 4]
            if coverage.min() < 0 or coverage.max() > 1:
                errors.append(f"coverage not in [0,1]: [{coverage.min():.3f}, {coverage.max():.3f}]")

    except Exception as e:
        errors.append(f"Forward pass failed: {str(e)}")

    if errors:
        for e in errors:
            print(f"❌ VALIDATE Error: {e}")
        return False

    if verbose:
        print(f"✅ validate_decoder PASSED")
        print(f"   Output shape: {predictions.shape}")
        print(f"   path_loss range: [{predictions[:, 0].min():.1f}, {predictions[:, 0].max():.1f}] dB")
        print(f"   rssi range: [{predictions[:, 3].min():.1f}, {predictions[:, 3].max():.1f}] dBm")
        print(f"   coverage range: [{predictions[:, 4].min():.3f}, {predictions[:, 4].max():.3f}]")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: RFDecoder")
    print("=" * 60)


    n_nodes = 500
    hidden_dim = 512
    embeddings = torch.randn(n_nodes, hidden_dim)


    print("\n📊 Testando RFDecoder básico...")
    decoder = RFDecoder(in_channels=hidden_dim)
    valid1 = validate_decoder(decoder, embeddings)


    print("\n📊 Testando PhysicsConstrainedDecoder...")
    decoder_physics = PhysicsConstrainedDecoder(in_channels=hidden_dim)
    valid2 = validate_decoder(decoder_physics, embeddings)


    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n   Decoder parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if (valid1 and valid2) else "❌ TESTE FALHOU")
    print("=" * 60)
