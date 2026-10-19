# GNN-RF Propagation V2
# 02_models/physics_loss.py

"""
Loss functions physics-informed para propagação RF.

Incorpora conhecimento de:
    - ITU-R P.833 (atenuação por vegetação)
    - ITU-R P.526 (difração)
    - Free Space Path Loss (limite inferior)
"""

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
    Loss function physics-informed para propagação RF.
    
    Combina:
        - MSE/Huber loss para predições
        - Constraints de ITU-R P.833 (vegetação)
        - Constraints de FSPL (limite inferior)
        - Learnable weights para balanceamento
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
        
        # Pesos aprendíveis para cada componente do loss
        if learnable_weights:
            # Log weights para garantir positividade
            self.log_weights = nn.Parameter(torch.zeros(5))
        else:
            self.register_buffer('log_weights', torch.zeros(5))
        
        # Índices dos targets
        self.target_names = [
            'path_loss_total',      # 0
            'path_loss_vegetation', # 1
            'path_loss_terrain',    # 2
            'rssi',                 # 3
            'coverage'              # 4
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
        Calcula loss total.
        
        Args:
            predictions: [N, 5] predições do modelo
            targets: [N, 5] targets reais
            canopy_height: [N] altura do dossel (para constraint P.833)
            distances: [N] distâncias Tx-Rx (para constraint FSPL)
        
        Returns:
            total_loss: Scalar
            loss_dict: Dict com componentes individuais
        """
        weights = self.weights
        loss_dict = {}
        
        # ==== Losses de predição ====
        for i, name in enumerate(self.target_names):
            pred = predictions[:, i]
            true = targets[:, i]
            
            if self.use_huber:
                loss = F.huber_loss(pred, true, delta=self.huber_delta, reduction='mean')
            else:
                loss = F.mse_loss(pred, true, reduction='mean')
            
            loss_dict[f'loss_{name}'] = loss
        
        # Weighted sum das losses de predição
        prediction_loss = sum(
            weights[i] * loss_dict[f'loss_{self.target_names[i]}']
            for i in range(5)
        )
        loss_dict['prediction_loss'] = prediction_loss
        
        # ==== Physics Constraints ====
        constraint_loss = torch.tensor(0.0, device=predictions.device)
        
        # Constraint ITU-R P.833 (vegetação)
        if canopy_height is not None:
            veg_constraint = self._vegetation_constraint(
                predictions[:, 1],  # path_loss_vegetation
                canopy_height
            )
            loss_dict['constraint_vegetation'] = veg_constraint
            constraint_loss = constraint_loss + self.vegetation_weight * veg_constraint
        
        # Constraint FSPL (limite inferior)
        if distances is not None:
            fspl_constraint = self._fspl_constraint(
                predictions[:, 0],  # path_loss_total
                distances
            )
            loss_dict['constraint_fspl'] = fspl_constraint
            constraint_loss = constraint_loss + self.fspl_weight * fspl_constraint
        
        loss_dict['constraint_loss'] = constraint_loss
        
        # ==== Total Loss ====
        total_loss = prediction_loss + constraint_loss
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def _vegetation_constraint(self,
                              pred_veg_loss: torch.Tensor,
                              canopy_height: torch.Tensor
                              ) -> torch.Tensor:
        """
        Constraint ITU-R P.833: perda por vegetação.
        
        A = d * (a * f^b)
        Onde: a ~ 0.2, b ~ 0.3 (floresta tropical)
        """
        # Coeficientes ITU-R P.833
        a = 0.2
        b = 0.3
        freq_ghz = self.frequency_mhz / 1000.0
        
        # Perda esperada (simplificada)
        # Assumindo path_length ~ canopy_height (vertical)
        expected_loss = a * (freq_ghz ** b) * torch.clamp(canopy_height, 0, 30)
        
        # Constraint: predição deve ser próxima do esperado
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
        
        # FSPL em dB
        d_km = distances / 1000.0 + 1e-6  # Evitar log(0)
        fspl = 20 * torch.log10(d_km) + 20 * math.log10(self.frequency_mhz) + 32.44
        
        # Constraint: pred_total >= FSPL (penalizar violações)
        violation = F.relu(fspl - pred_total_loss)
        
        return violation.mean()


class CurriculumRFLoss(PhysicsRFLoss):
    """
    Loss com suporte a curriculum learning.

    Componentes de penalidade fisica:
      - distance_gradient: penaliza pares (i,j) onde dist_i < dist_j mas RSSI_i < RSSI_j
      - variance:          penaliza quando rssi_std_pred < rssi_std_target (suavizacao)
      - shadowing_ndvi:    penaliza quando corr(shadow_pred, ndvi) < min_corr (fisica errada)
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

        # Máscaras de ativação por target
        self.register_buffer('target_mask', torch.ones(5))

        # Peso dos constraints por fase
        self.constraint_scale = 1.0
    
    def _distance_gradient_penalty(
        self,
        pred_rssi: torch.Tensor,     # (N,) RSSI previsto
        dist_to_ant: torch.Tensor,   # (N,) distancia ao transmissor (metros)
    ) -> torch.Tensor:
        """
        Penalidade de gradiente de distancia.

        Amostra K pares aleatorios (i, j) dentro do batch.
        Para cada par onde dist_i < dist_j (i esta mais proximo),
        penaliza se RSSI_i <= RSSI_j (RSSI deveria ser maior perto da antena).

        Penalty = mean( ReLU(rssi_j - rssi_i + margin) )  para todos pares
                  onde dist_i < dist_j

        margin = 1 dBm — folga para nao penalizar diferencas triviais.
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

        # Mascara: i esta mais proximo que j
        closer = dist_i < dist_j
        if closer.sum() == 0:
            return torch.tensor(0.0, device=pred_rssi.device)

        margin = 1.0  # dBm
        # Se i esta mais perto, rssi_i deve ser > rssi_j
        # Violacao: rssi_j - rssi_i + margin > 0
        violation = torch.relu(rssi_j[closer] - rssi_i[closer] + margin)
        return violation.mean()

    # ------------------------------------------------------------------
    # Penalidade de variancia (Acao 3 v8)
    # ------------------------------------------------------------------

    def _variance_penalty(
        self,
        pred_rssi: torch.Tensor,   # (N,) RSSI previsto
        tgt_rssi: torch.Tensor,    # (N,) RSSI alvo
    ) -> torch.Tensor:
        """
        Penaliza quando std das predicoes e menor que std dos targets.

        O modelo tende ao valor medio (regressao para media) ao minimizar Huber.
        Isso comprime a distribuicao de RSSI, fazendo todos os nos passarem
        no limiar de cobertura. Esta penalidade forca o modelo a manter a
        variabilidade dos targets.

        Retorna relu(tgt_std - pred_std)^2:
          zero quando pred_std >= tgt_std (correto ou sobre-disperso)
          positivo quando pred_std < tgt_std (suavizacao excessiva)
        """
        pred_std = pred_rssi.std()
        tgt_std  = tgt_rssi.std().detach()   # sem gradiente nos targets
        if tgt_std < 1e-4:
            return torch.tensor(0.0, device=pred_rssi.device)
        return torch.relu(tgt_std - pred_std) ** 2

    # ------------------------------------------------------------------
    # Penalidade de correlacao shadow-NDVI (Acao 4 v8)
    # ------------------------------------------------------------------

    def _shadowing_ndvi_penalty(
        self,
        shadow_pred: torch.Tensor,   # (N,) shadow_margin previsto
        ndvi_feat: torch.Tensor,     # (N,) NDVI do no (feature col 12), ja no device
    ) -> torch.Tensor:
        """
        Penaliza quando shadow_pred nao correlaciona positivamente com NDVI.

        Fisica esperada: shadow_margin cresce com densidade de vegetacao (NDVI alto).
        corr(shadow_pred, ndvi) deve ser > min_corr.

        Implementacao diferenciavel: gradiente propaga por shadow_pred,
        ndvi_feat e tratado como constante (detach).
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
        Configura loss para fase específica do curriculum.
        
        Fases:
            1: Foundation - apenas path_loss_total
            2: Terrain - + path_loss_terrain
            3: Vegetation - + path_loss_vegetation
            4: RSSI - + rssi
            5: Coverage - todos + fine-tuning
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
        Forward com mascara de curriculum e penalidades de fisica.

        Args extras:
            dist_to_ant : (N,) distancia em metros ate a antena mais proxima.
                          Ativa penalidade de gradiente de distancia no RSSI (col 3).
            ndvi        : (N,) NDVI de cada no (feature col 12).
                          Ativa penalidade de correlacao shadow-NDVI quando shadow
                          esta ativo na mascara (fase >= 3).
        """
        total_loss, loss_dict = super().forward(predictions, targets, **kwargs)

        # Aplicar máscara de curriculum
        masked_loss = torch.tensor(0.0, device=predictions.device)
        for i, name in enumerate(self.target_names):
            if self.target_mask[i] > 0:
                masked_loss = masked_loss + self.weights[i] * loss_dict[f'loss_{name}']

        # Escalar constraints fisicos
        if 'constraint_loss' in loss_dict:
            masked_loss = masked_loss + self.constraint_scale * loss_dict['constraint_loss']

        pred_rssi = predictions[:, 3]   # col 3 = rssi

        # Penalidade de gradiente de distancia (v7)
        dist_pen = torch.tensor(0.0, device=predictions.device)
        if dist_to_ant is not None and self.distance_gradient_weight > 0:
            dist_pen = self._distance_gradient_penalty(pred_rssi, dist_to_ant)
            masked_loss = masked_loss + self.distance_gradient_weight * dist_pen

        # Penalidade de variancia RSSI (Acao 3 v8)
        var_pen = torch.tensor(0.0, device=predictions.device)
        if self.variance_weight > 0:
            var_pen = self._variance_penalty(pred_rssi, targets[:, 3])
            masked_loss = masked_loss + self.variance_weight * var_pen

        # Penalidade de correlacao shadow-NDVI (Acao 4 v8) — apenas fase >= 3
        ndvi_pen = torch.tensor(0.0, device=predictions.device)
        shadow_active = (self.target_mask[1] > 0)   # col 1 = shadow_margin
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
    Valida loss function.
    
    Returns:
        bool: True se válido
    """
    errors = []
    
    try:
        # Dados sintéticos
        n = 100
        predictions = torch.randn(n, 5)
        targets = torch.randn(n, 5)
        canopy_height = torch.rand(n) * 30
        distances = torch.rand(n) * 10000
        
        # Calcular loss
        total_loss, loss_dict = loss_fn(
            predictions, targets,
            canopy_height=canopy_height,
            distances=distances
        )
        
        # Verificar que loss é escalar
        if total_loss.dim() != 0:
            errors.append(f"Loss should be scalar, got shape {total_loss.shape}")
        
        # Verificar que loss não é NaN
        if torch.isnan(total_loss):
            errors.append("Loss is NaN")
        
        # Verificar que loss é finito
        if torch.isinf(total_loss):
            errors.append("Loss is infinite")
        
        # Verificar componentes
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
    
    # Testar PhysicsRFLoss
    print("\n📊 Testando PhysicsRFLoss...")
    loss_fn = PhysicsRFLoss(frequency_mhz=900.0)
    valid1 = validate_physics_loss(loss_fn)
    
    # Testar CurriculumRFLoss
    print("\n📊 Testando CurriculumRFLoss...")
    curriculum_loss = CurriculumRFLoss(frequency_mhz=1800.0)
    
    for phase in [1, 2, 3, 4, 5]:
        curriculum_loss.set_phase(phase)
        print(f"   Phase {phase}: mask={curriculum_loss.target_mask.tolist()}, scale={curriculum_loss.constraint_scale}")
    
    valid2 = validate_physics_loss(curriculum_loss)
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO" if (valid1 and valid2) else "❌ TESTE FALHOU")
    print("=" * 60)
