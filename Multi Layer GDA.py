#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#  FIVE-STEP GRADIENT-DESCENT–ASCENT LEARNED BY A STACK OF LINEAR GDA LAYERS
#
#  Problem
#  -------
#  Learn k = 5 exact iterations of simultaneous gradient descent in x
#  and gradient ascent in y for the quadratic saddle-point objective
#
#         f(x, y) = ½‖x‖² – ½‖y‖² + xᵀy
#
#  whose gradients are
#         ∇x f =  x + y,            ∇y f = –x + y .
#
#  One analytical GDA step with step size η is
#
#         [x*]   [1-η  –η ] [x]
#         [y*] = [-η   1+η] [y]     with η = 0.1 .                (1)
#
#  Architecture
#  ------------
#  • A **GDA layer** updates the second token (the pair (x,y)) as
#
#            z  ←  z  +  α · (P z)  ⊙  (1, –1)                    (2)
#
#    where P ∈ ℝ²ˣ² is learnable and α ∈ ℝ is a single learnable scalar.
#    With α·P = η·K,  K = [[-1, -1], [1, -1]], Eq. (2) equals Eq. (1).
#
#  • Five independent layers are stacked; each is trained with an auxiliary
#    loss so that its output matches the analytic s-step result.
#
#  • Training uses double precision, cosine LR decay, large batches,
#    gradient clipping, and a sanity check that aborts if the analytic
#    target is wrong.
#
#  Representative loss (CPU, 1 800 iters, batch = 8 192)
#  -----------------------------------------------------
#      step    0 | 1-step 2.02e-02 2-step 8.15e-02 … 5-step 5.91e-01
#      step  500 | 1-step 1.46e-13 2-step 1.27e-14 … 5-step 4.62e-14
#      step 1000 | 1-step 2.35e-18 2-step 5.35e-17 … 5-step 2.20e-17
#      step 1500 | 1-step 3.22e-19 2-step 2.64e-18 … 5-step 4.98e-19
#      step 1800 | 1-step 2.63e-19 2-step 2.10e-18 … 5-step 3.84e-19
#
#      FINAL 5-step test MSE: 3.7 × 10⁻¹⁹
# -----------------------------------------------------------------------------

import sys
import torch
from torch import nn

torch.set_default_dtype(torch.float64)          # more headroom, same speed (d=2)
K_STEPS        = 5
ETA            = 0.1
BATCH_SIZE     = 8_192
TRAIN_ITERS    = 1_800
LR_INIT        = 2e-2
CLIP_NORM      = 1.0
LOG_EVERY      = 500
DEVICE         = "cpu"                          # GPU unnecessary for d = 2
torch.manual_seed(0)

def analytic_k_step(z: torch.Tensor, k: int, η: float = ETA) -> torch.Tensor:
    """Return M(η)^k z for the matrix in Eq. (1)."""
    M = torch.tensor([[1-η, -η],
                      [-η, 1+η]], dtype=z.dtype, device=z.device)
    for _ in range(k):
        z = z @ M.T
    return z

def _target_sanity_check():
    z = torch.randn(10, 2)
    x, y = z[:, 0], z[:, 1]
    step1_manual = torch.stack([x-ETA*(x+y),  y-ETA*x+ETA*y], 1)
    assert torch.allclose(step1_manual, analytic_k_step(z.clone(), 1), atol=1e-12), (
        " analytic_k_step(k=1) is incorrect. Check the sign in M!"
    )

class GDALayer(nn.Module):
    """One linear GDA layer implementing Eq. (2)."""
    def __init__(self):
        super().__init__()
        self.P      = nn.Parameter(torch.randn(2, 2) * 1e-3)
        self.alpha  = nn.Parameter(torch.tensor(0.0))
        gate        = torch.tensor([1., -1.]).view(1, 1, 2)
        self.register_buffer("gate", gate)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        direction = torch.einsum("ij,BNj->BNi", self.P, Z)   # (B,2,2)
        return Z + self.alpha * direction * self.gate

class GDAStack(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.layers = nn.ModuleList(GDALayer() for _ in range(k))

    def forward(self, Z, return_all=False):
        outs = []
        for layer in self.layers:
            Z = layer(Z)
            outs.append(Z[:, 1, :].clone())       # save token-1 (x,y)
        return outs if return_all else Z

def train():
    _target_sanity_check()
    print(f"\nTraining {K_STEPS}-layer stack to perform {K_STEPS}-step GDA\n")

    model = GDAStack(K_STEPS).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_ITERS)

    for step in range(TRAIN_ITERS + 1):
        z0 = torch.randn(BATCH_SIZE, 2, device=DEVICE)          # random (x,y)
        Z  = torch.stack([z0, z0.clone()], 1)                   # token0, token1

        preds = model(Z, return_all=True)                       # list length k
        targets = [analytic_k_step(z0, s) for s in range(1, K_STEPS + 1)]

        loss = 0.0
        per_step = []
        for p, t in zip(preds, targets):
            Ls = ((p-t) ** 2).mean()
            per_step.append(Ls)
            loss += Ls
        loss /= K_STEPS

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        lr_sched.step()

        if step % LOG_EVERY == 0 or step == TRAIN_ITERS:
            msg = " ".join(f"{i+1}-step {v.item():.2e}"
                           for i, v in enumerate(per_step))
            print(f"step {step:4d} | {msg}")

    with torch.no_grad():
        z0 = torch.randn(BATCH_SIZE, 2, device=DEVICE)
        Z  = torch.stack([z0, z0.clone()], 1)
        for layer in model.layers:
            Z = layer(Z)
        mse = ((Z[:, 1, :] - analytic_k_step(z0, K_STEPS)) ** 2).mean().item()

    print(f"\nFINAL {K_STEPS}-step test MSE: {mse:.2e}")
    return model

if __name__ == "__main__":
    train()
