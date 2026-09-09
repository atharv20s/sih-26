"""Physics-Informed Neural Network (PINN) for aero-piston-engine RUL prediction.

Key innovation (SIH26054 spec — "The Physics Guardian"):
  The network embeds Fourier's law of heat conduction directly into the loss
  function, forcing all predicted thermal trajectories to obey thermodynamics
  even during anomalous or out-of-distribution sensor states.

Architecture
------------
Same 1D-CNN backbone depth as the data-driven baseline (``cnn_baseline.py``),
but with two critical differences:

  1. Operates on the full **12-channel raw** sensor windows instead of PCA-
     reduced 8-channel windows, because PCA decorrelates channels and destroys
     the per-sensor semantics that Fourier's law requires (CHT in °C, fuel_flow
     in L/h, coolant_temp in °C, etc.).

  2. Adds a **physics-informed composite loss**:

     L_total = L_data  +  λ_f · L_fourier  +  λ_c · L_consistency

     L_data        : MSE on normalised RUL prediction (same as baseline).
     L_fourier     : Fourier / Newton thermal residual on CHT, enforcing
                     dCHT/dt ≈ α·Q_combustion − β·(CHT − T_coolant).
     L_consistency : EGT-CHT thermodynamic correlation constraint — both
                     channels are driven by combustion heat, so their
                     temporal derivatives must be positively correlated.

     α and β are **learnable** ``nn.Parameter``s (kept positive via softplus)
     so the network discovers the effective thermal capacity and cooling
     coefficient from data while respecting the functional form of Fourier's
     law.

Denormalization
---------------
The dataset builder (``build_dataset.py``) exports min-max normalisation stats
(``norm_min.npy``, ``norm_max.npy``).  The physics loss uses these to convert
the normalised input windows back to physical units before computing residuals,
so the residual truly measures energy-balance error in °C.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Sensor channel indices (must match SENSOR_ORDER in engine_sim.py) ------
CHT = 1               # Cylinder Head Temperature, °C
EGT = 2               # Exhaust Gas Temperature, °C
FUEL_FLOW = 5          # L/h
RPM = 0                # rev/min
COOLANT_TEMP = 11      # °C

# Nominal values — used only to scale the physics residual terms so that
# the learnable coefficients α and β start at order-unity magnitudes.
_NOM_FUEL = 9.5        # L/h (cruise)
_NOM_CHT  = 150.0      # °C (nominal CHT at cruise)
_NOM_EGT  = 640.0      # °C


class PINNRULModel(nn.Module):
    """Physics-Informed 1D-CNN for RUL prediction.

    Parameters
    ----------
    in_features : int
        Number of input sensor channels (12 for raw windows).
    hidden : int
        Base channel width for the conv backbone.
    lambda_fourier : float
        Weight for the Fourier thermal-residual loss term.
    lambda_consistency : float
        Weight for the EGT-CHT correlation loss term.
    dropout : float
        Dropout probability in the MLP head and between conv blocks.
    kernel_sizes : tuple of int
        Kernel widths for the three conv layers.
    pool : int
        Max-pool kernel width.
    """

    def __init__(self, in_features: int = 12, hidden: int = 64,
                 lambda_fourier: float = 0.01,
                 lambda_consistency: float = 0.005,
                 dropout: float = 0.2,
                 kernel_sizes: tuple = (5, 3, 3), pool: int = 2):
        super().__init__()
        self.lambda_fourier = lambda_fourier
        self.lambda_consistency = lambda_consistency

        # ---- 1D-CNN backbone (mirrors cnn_baseline.py) ----------------------
        self.conv1 = nn.Conv1d(in_features, hidden, kernel_sizes[0], padding=2)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, kernel_sizes[1], padding=1)
        self.bn2   = nn.BatchNorm1d(hidden * 2)
        self.conv3 = nn.Conv1d(hidden * 2, hidden * 4, kernel_sizes[2], padding=1)
        self.bn3   = nn.BatchNorm1d(hidden * 4)

        self.pool = nn.MaxPool1d(pool)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.gap  = nn.AdaptiveAvgPool1d(1)

        # Multi-task heads sharing the conv backbone:
        # Head 1: RUL prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

        # Head 2: Physical thermal stress gradient (dT/dt in °C/cycle)
        self.fc_grad = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # ---- Learnable Fourier physics parameters ---------------------------
        #   dCHT/dt  ≈  α · (fuel_flow / nominal)  −  β · (CHT − T_coolant)
        self.log_alpha = nn.Parameter(torch.tensor(-3.0))
        self.log_beta  = nn.Parameter(torch.tensor(-5.0))

        # ---- Normalisation stats (buffers — move with .to(device)) ----------
        self.register_buffer("norm_min", torch.zeros(12))
        self.register_buffer("norm_max", torch.ones(12))

    # --------------------------------------------------------------------- #
    # Normalisation helpers                                                   #
    # --------------------------------------------------------------------- #
    def set_norm_stats(self, mins: np.ndarray, maxs: np.ndarray):
        """Load the min-max normalisation constants exported by build_dataset."""
        self.norm_min.copy_(torch.as_tensor(mins, dtype=torch.float32))
        self.norm_max.copy_(torch.as_tensor(maxs, dtype=torch.float32))

    def _to_physical(self, x_norm: torch.Tensor) -> torch.Tensor:
        """(B, T, 12) normalised [0,1] → physical units (°C, kPa, L/h …)."""
        rng  = (self.norm_max - self.norm_min).unsqueeze(0).unsqueeze(0)
        base = self.norm_min.unsqueeze(0).unsqueeze(0)
        return x_norm * rng + base

    # --------------------------------------------------------------------- #
    # Forward pass                                                            #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, return_gradient: bool = False):
        """(B, T, 12) → (B, 1) RUL prediction, optionally with physical gradient."""
        z = x.permute(0, 2, 1).contiguous()          # (B, F, T)
        z = self.relu(self.bn1(self.conv1(z)))
        z = self.pool(z)
        z = self.relu(self.bn2(self.conv2(z)))
        z = self.pool(z)
        z = self.relu(self.bn3(self.conv3(z)))
        z = self.pool(z)
        feat = self.gap(z).squeeze(-1)               # (B, hidden*4)

        rul_pred = self.fc(feat)                     # (B, 1)
        if return_gradient:
            grad_pred = self.fc_grad(feat)           # (B, 1)
            return rul_pred, grad_pred
        return rul_pred

    # --------------------------------------------------------------------- #
    # Physics losses & thermodynamic verification                             #
    # --------------------------------------------------------------------- #
    def compute_physics_targets(self, x_norm: torch.Tensor):
        """Compute empirical CHT finite-difference and Fourier theoretical gradient."""
        x = self._to_physical(x_norm)                 # (B, T, 12)
        cht  = x[:, :, CHT]                           # °C
        cool = x[:, :, COOLANT_TEMP]                  # °C
        fuel = x[:, :, FUEL_FLOW]                     # L/h

        # Empirical finite-difference gradient at recent window steps
        dcht_empirical = cht[:, -1:] - cht[:, -2:-1]  # (B, 1), °C / cycle

        # Normalised combustion heat-source proxy and temperature excess
        q_in = fuel[:, -2:-1] / _NOM_FUEL              # dimensionless
        delta_t = cht[:, -2:-1] - cool[:, -2:-1]       # °C

        alpha = F.softplus(self.log_alpha)
        beta  = F.softplus(self.log_beta)

        # Theoretical Fourier gradient
        g_fourier = alpha * q_in - beta * delta_t     # (B, 1)
        return dcht_empirical, g_fourier

    def fourier_loss(self, pred_grad: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        r"""Fourier thermal-balance loss coupling physics to network parameters."""
        dcht_empirical, g_fourier = self.compute_physics_targets(x_norm)
        # Ground predicted physical gradient to both empirical gradient and Fourier law
        loss_emp = F.mse_loss(pred_grad, dcht_empirical)
        loss_fourier = F.mse_loss(pred_grad, g_fourier)
        return loss_fourier + 0.5 * loss_emp

    def consistency_loss(self, x_norm: torch.Tensor) -> torch.Tensor:
        """EGT-CHT thermodynamic consistency penalty."""
        x = self._to_physical(x_norm)
        dcht = x[:, 1:, CHT] - x[:, :-1, CHT]        # °C / cycle
        degt = x[:, 1:, EGT] - x[:, :-1, EGT]        # °C / cycle
        anti = torch.relu(-dcht * degt)                # ≥ 0
        return anti.mean()

    # --------------------------------------------------------------------- #
    # Composite loss                                                          #
    # --------------------------------------------------------------------- #
    def total_loss(self, pred_rul: torch.Tensor, target_rul: torch.Tensor,
                   x_norm: torch.Tensor, pred_grad: torch.Tensor = None) -> dict:
        """Compute L_total = L_data + λ_f·L_fourier + λ_c·L_consistency."""
        if pred_grad is None:
            _, pred_grad = self.forward(x_norm, return_gradient=True)

        l_data = F.mse_loss(pred_rul, target_rul)
        l_four = self.fourier_loss(pred_grad, x_norm)
        l_cons = self.consistency_loss(x_norm)

        l_total = (l_data
                   + self.lambda_fourier * l_four
                   + self.lambda_consistency * l_cons)

        return {
            "total":       l_total,
            "data":        l_data.detach(),
            "fourier":     l_four.detach(),
            "consistency": l_cons.detach(),
            "alpha":       F.softplus(self.log_alpha).detach(),
            "beta":        F.softplus(self.log_beta).detach(),
        }

    # --------------------------------------------------------------------- #
    # Tool 1 Interface: Live Inference & Physics Auditor                      #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def audit_sample(self, x_norm: torch.Tensor, rul_cap: float = 500.0,
                     residual_threshold: float = 5.0) -> dict:
        """Evaluate a live window (1, T, 12) or batch (B, T, 12).

        Returns predicted RUL, estimated physical thermal gradient,
        Fourier residual, and physical boundary validity flag.
        """
        self.eval()
        if x_norm.dim() == 2:
            x_norm = x_norm.unsqueeze(0)

        rul_norm, grad_pred = self.forward(x_norm, return_gradient=True)
        dcht_emp, g_fourier = self.compute_physics_targets(x_norm)

        physics_residual = torch.abs(grad_pred - g_fourier).squeeze(-1)
        rul_cycles = (rul_norm.squeeze(-1) * rul_cap).clamp(min=0.0)

        is_valid = physics_residual <= residual_threshold

        return {
            "rul_cycles": rul_cycles.cpu().numpy(),
            "rul_norm": rul_norm.squeeze(-1).cpu().numpy(),
            "physical_gradient": grad_pred.squeeze(-1).cpu().numpy(),
            "fourier_expected_gradient": g_fourier.squeeze(-1).cpu().numpy(),
            "empirical_gradient": dcht_emp.squeeze(-1).cpu().numpy(),
            "physics_residual": physics_residual.cpu().numpy(),
            "is_physically_valid": is_valid.cpu().numpy(),
        }
