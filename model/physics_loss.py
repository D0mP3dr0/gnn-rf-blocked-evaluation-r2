

"""Physics-informed loss functions for RF propagation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


class PhysicsRFLoss(nn.Module):
    """
    Physics-informed loss function for RF propagation.

    Combines:
        - MSE/Huber loss on predictions.
        - ITU-R P.833 vegetation constraints.
        - FSPL constraint (lower bound).
        - Learnable weights for balancing terms.
    """

    def __init__(self,
                 frequency_mhz: float = 900.0,
                 use_huber: bool = True,
                 huber_delta: float = 5.0,
                 vegetation_weight: float = 0.1,
                 fspl_weight: float = 0.05,
                 learnable_weights: bool = True):
        super().__init__()

        self.frequency_mhz = frequency_mhz
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.vegetation_weight = vegetation_weight
        self.fspl_weight = fspl_weight


        if learnable_weights:

            self.log_weights = nn.Parameter(torch.zeros(5))
        else:
            self.register_buffer('log_weights', torch.zeros(5))


        self.target_names = [
            'path_loss_total',
            'path_loss_vegetation',
            'path_loss_terrain',
            'rssi',
            'coverage'
        ]

    @property
    def weights(self) -> torch.Tensor:
        """Retorna pesos normalizados (softmax)."""
        return F.softmax(self.log_weights, dim=0)

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                canopy_height: Optional[torch.Tensor] = None,
                distances: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the total loss.

        Args:
            predictions: [N, 5] model predictions.
            targets: [N, 5] ground-truth targets.
            canopy_height: [N] canopy height (for the P.833 constraint).
            distances: [N] Tx-Rx distances (for the FSPL constraint).

        Returns:
            total_loss: scalar.
            loss_dict: dict of individual components.
        """
        weights = self.weights
        loss_dict = {}


        for i, name in enumerate(self.target_names):
            pred = predictions[:, i]
            true = targets[:, i]

            if self.use_huber:
                loss = F.huber_loss(pred, true, delta=self.huber_delta, reduction='mean')
            else:
                loss = F.mse_loss(pred, true, reduction='mean')

            loss_dict[f'loss_{name}'] = loss


        prediction_loss = sum(
            weights[i] * loss_dict[f'loss_{self.target_names[i]}']
            for i in range(5)
        )
        loss_dict['prediction_loss'] = prediction_loss


        constraint_loss = torch.tensor(0.0, device=predictions.device)


        if canopy_height is not None:
            veg_constraint = self._vegetation_constraint(
                predictions[:, 1],
                canopy_height
            )
            loss_dict['constraint_vegetation'] = veg_constraint
            constraint_loss = constraint_loss + self.vegetation_weight * veg_constraint


        if distances is not None:
            fspl_constraint = self._fspl_constraint(
                predictions[:, 0],
                distances
            )
            loss_dict['constraint_fspl'] = fspl_constraint
            constraint_loss = constraint_loss + self.fspl_weight * fspl_constraint

        loss_dict['constraint_loss'] = constraint_loss


        total_loss = prediction_loss + constraint_loss
        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict

    def _vegetation_constraint(self,
                              pred_veg_loss: torch.Tensor,
                              canopy_height: torch.Tensor
                              ) -> torch.Tensor:
        """
        ITU-R P.833 constraint: vegetation loss.

        A = d * (a * f^b)
        with a ~ 0.2, b ~ 0.3 (tropical forest).
        """

        a = 0.2
        b = 0.3
        freq_ghz = self.frequency_mhz / 1000.0


        # Assumes path_length ~ canopy_height (vertical)
        expected_loss = a * (freq_ghz ** b) * torch.clamp(canopy_height, 0, 30)


        constraint = F.huber_loss(pred_veg_loss, expected_loss, delta=3.0)

        return constraint

    def _fspl_constraint(self,
                        pred_total_loss: torch.Tensor,
                        distances: torch.Tensor
                        ) -> torch.Tensor:
        """
        Constraint FSPL: perda total >= Free Space Path Loss.

        FSPL = 20*log10(d) + 20*log10(f) + 32.44
        """
        import math

        # FSPL in dB
        d_km = distances / 1000.0 + 1e-6
        fspl = 20 * torch.log10(d_km) + 20 * math.log10(self.frequency_mhz) + 32.44


        violation = F.relu(fspl - pred_total_loss)

        return violation.mean()


class CurriculumRFLoss(PhysicsRFLoss):
    """
    Loss with curriculum-learning support.

    Physics penalty components:
      - distance_gradient: penalizes pairs (i, j) where dist_i < dist_j but RSSI_i < RSSI_j.
      - variance:          penalizes rssi_std_pred < rssi_std_target (over-smoothing).
      - shadowing_ndvi:    penalizes corr(shadow_pred, ndvi) below min_corr (wrong physics).
    """

    def __init__(
        self,
        distance_gradient_weight: float = 0.05,
        distance_gradient_n_pairs: int = 512,
        variance_weight: float = 0.02,
        shadowing_ndvi_weight: float = 0.03,
        shadowing_ndvi_min_corr: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.distance_gradient_weight  = distance_gradient_weight
        self.distance_gradient_n_pairs = distance_gradient_n_pairs
        self.variance_weight           = variance_weight
        self.shadowing_ndvi_weight     = shadowing_ndvi_weight
        self.shadowing_ndvi_min_corr   = shadowing_ndvi_min_corr


        self.register_buffer('target_mask', torch.ones(5))


        self.constraint_scale = 1.0

    def _distance_gradient_penalty(
        self,
        pred_rssi: torch.Tensor,
        dist_to_ant: torch.Tensor,   # (N,) distance to transmitter (meters)
    ) -> torch.Tensor:
        """
        Distance-gradient penalty.

        Samples K random pairs (i, j) within the batch. For each pair where
        dist_i < dist_j (i is closer), penalizes RSSI_i <= RSSI_j (RSSI should
        be higher closer to the antenna).

        penalty = mean(ReLU(rssi_j - rssi_i + margin)) over all pairs with
        dist_i < dist_j; margin = 1 dBm, to avoid penalizing trivial
        differences.
        """
        N = pred_rssi.shape[0]
        K = min(self.distance_gradient_n_pairs, N // 2)
        if K < 4:
            return torch.tensor(0.0, device=pred_rssi.device)

        idx = torch.randperm(N, device=pred_rssi.device)[:K * 2]
        i_idx = idx[:K]
        j_idx = idx[K:]

        dist_i = dist_to_ant[i_idx]
        dist_j = dist_to_ant[j_idx]
        rssi_i = pred_rssi[i_idx]
        rssi_j = pred_rssi[j_idx]


        closer = dist_i < dist_j
        if closer.sum() == 0:
            return torch.tensor(0.0, device=pred_rssi.device)

        margin = 1.0  # dBm


        violation = torch.relu(rssi_j[closer] - rssi_i[closer] + margin)
        return violation.mean()


    def _variance_penalty(
        self,
        pred_rssi: torch.Tensor,
        tgt_rssi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalizes prediction std lower than target std.

        Minimizing the Huber loss biases the model toward the mean, compressing
        the predicted RSSI distribution so that most nodes end up on the same
        side of the coverage threshold. This penalty keeps the model from
        collapsing the target variability.

        Returns relu(tgt_std - pred_std)^2: zero when pred_std >= tgt_std
        (correct or over-dispersed), positive when pred_std < tgt_std
        (over-smoothing).
        """
        pred_std = pred_rssi.std()
        tgt_std  = tgt_rssi.std().detach()
        if tgt_std < 1e-4:
            return torch.tensor(0.0, device=pred_rssi.device)
        return torch.relu(tgt_std - pred_std) ** 2


    def _shadowing_ndvi_penalty(
        self,
        shadow_pred: torch.Tensor,
        ndvi_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalizes shadow_pred not correlating positively with NDVI.

        Expected physics: shadow_margin should grow with vegetation density
        (high NDVI); corr(shadow_pred, ndvi) should exceed min_corr.
        Differentiable: gradients flow through shadow_pred only, ndvi_feat is
        treated as a constant (detached).
        """
        n = shadow_pred.shape[0]
        if n < 10:
            return torch.tensor(0.0, device=shadow_pred.device)

        nf = ndvi_feat.detach().float()
        if nf.std() < 1e-6:
            return torch.tensor(0.0, device=shadow_pred.device)

        sp = shadow_pred - shadow_pred.mean()
        nf_c = nf - nf.mean()
        std_s = shadow_pred.std().clamp(min=1e-6)
        std_n = nf.std().clamp(min=1e-6)
        corr  = (sp * nf_c).mean() / (std_s * std_n)
        return torch.relu(self.shadowing_ndvi_min_corr - corr)

    def set_phase(self, phase: int) -> None:
        """
        Configures the loss for a given curriculum phase.

        Phases:
            1: Foundation - path_loss_total only.
            2: Terrain - adds path_loss_terrain.
            3: Vegetation - adds path_loss_vegetation.
            4: RSSI - adds rssi.
            5: Coverage - all terms, fine-tuning.
        """
        if phase == 1:
            self.target_mask = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
            self.constraint_scale = 0.0
        elif phase == 2:
            self.target_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
            self.constraint_scale = 0.0
        elif phase == 3:
            self.target_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
            self.constraint_scale = 0.5
        elif phase == 4:
            self.target_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
            self.constraint_scale = 0.8
        else:  # phase >= 5
            self.target_mask = torch.ones(5)
            self.constraint_scale = 1.0

    def forward(
        self,
        predictions,
        targets,
        dist_to_ant: Optional[torch.Tensor] = None,
        ndvi: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with the curriculum mask and physics penalties.

        Extra args:
            dist_to_ant: (N,) distance in meters to the nearest antenna;
                enables the distance-gradient penalty on RSSI (column 3).
            ndvi: (N,) NDVI per node (feature column 12); enables the
                shadow-NDVI correlation penalty once shadow is active in the
                mask (phase >= 3).
        """
        total_loss, loss_dict = super().forward(predictions, targets, **kwargs)


        masked_loss = torch.tensor(0.0, device=predictions.device)
        for i, name in enumerate(self.target_names):
            if self.target_mask[i] > 0:
                masked_loss = masked_loss + self.weights[i] * loss_dict[f'loss_{name}']


        if 'constraint_loss' in loss_dict:
            masked_loss = masked_loss + self.constraint_scale * loss_dict['constraint_loss']

        pred_rssi = predictions[:, 3]


        dist_pen = torch.tensor(0.0, device=predictions.device)
        if dist_to_ant is not None and self.distance_gradient_weight > 0:
            dist_pen = self._distance_gradient_penalty(pred_rssi, dist_to_ant)
            masked_loss = masked_loss + self.distance_gradient_weight * dist_pen


        var_pen = torch.tensor(0.0, device=predictions.device)
        if self.variance_weight > 0:
            var_pen = self._variance_penalty(pred_rssi, targets[:, 3])
            masked_loss = masked_loss + self.variance_weight * var_pen


        ndvi_pen = torch.tensor(0.0, device=predictions.device)
        shadow_active = (self.target_mask[1] > 0)
        if ndvi is not None and shadow_active and self.shadowing_ndvi_weight > 0:
            shadow_pred = predictions[:, 1]
            ndvi_pen    = self._shadowing_ndvi_penalty(shadow_pred, ndvi)
            masked_loss = masked_loss + self.shadowing_ndvi_weight * ndvi_pen

        loss_dict['dist_gradient_penalty'] = dist_pen
        loss_dict['variance_penalty']      = var_pen
        loss_dict['shadowing_ndvi_penalty'] = ndvi_pen
        loss_dict['masked_loss']           = masked_loss

        return masked_loss, loss_dict


def validate_physics_loss(loss_fn: PhysicsRFLoss,
                         verbose: bool = True) -> bool:
    """
    Validates the loss function.

    Returns:
        bool: True if valid.
    """
    errors = []

    try:

        n = 100
        predictions = torch.randn(n, 5)
        targets = torch.randn(n, 5)
        canopy_height = torch.rand(n) * 30
        distances = torch.rand(n) * 10000


        total_loss, loss_dict = loss_fn(
            predictions, targets,
            canopy_height=canopy_height,
            distances=distances
        )


        if total_loss.dim() != 0:
            errors.append(f"Loss should be scalar, got shape {total_loss.shape}")


        if torch.isnan(total_loss):
            errors.append("Loss is NaN")


        if torch.isinf(total_loss):
            errors.append("Loss is infinite")


        required_keys = ['total_loss', 'prediction_loss']
        for key in required_keys:
            if key not in loss_dict:
                errors.append(f"Missing key: {key}")

    except Exception as e:
        errors.append(f"Loss computation failed: {str(e)}")

    if errors:
        for e in errors:
            print(f"❌ VALIDATE Error: {e}")
        return False

    if verbose:
        print(f"✅ validate_physics_loss PASSED")
        print(f"   Total loss: {total_loss.item():.4f}")
        if 'constraint_vegetation' in loss_dict:
            print(f"   Vegetation constraint: {loss_dict['constraint_vegetation'].item():.4f}")
        if 'constraint_fspl' in loss_dict:
            print(f"   FSPL constraint: {loss_dict['constraint_fspl'].item():.4f}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE: PhysicsRFLoss")
    print("=" * 60)


    print("\n📊 Testando PhysicsRFLoss...")
    loss_fn = PhysicsRFLoss(frequency_mhz=900.0)
    valid1 = validate_physics_loss(loss_fn)


    print("\n📊 Testando CurriculumRFLoss...")
    curriculum_loss = CurriculumRFLoss(frequency_mhz=1800.0)

    for phase in [1, 2, 3, 4, 5]:
        curriculum_loss.set_phase(phase)
        print(f"   Phase {phase}: mask={curriculum_loss.target_mask.tolist()}, scale={curriculum_loss.constraint_scale}")

    valid2 = validate_physics_loss(curriculum_loss)

    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if (valid1 and valid2) else "❌ TESTE FALHOU")
    print("=" * 60)
