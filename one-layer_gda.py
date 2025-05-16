"""
The single-layer Transformer-like module here is learning the one-step 
gradient descent/ascent (GDA) update for the quadratic function:

    f(x, y) = 0.5 * ||x||^2 - 0.5 * ||y||^2 + x^T A y    with A=1.

Single-step GDA means:
    x -> x - eta * (df/dx),    y -> y + eta * (df/dy),
where df/dx = x + y, df/dy = -(x - y), and eta=0.1.

INSIDE THE LAYER
----------------
• The forward pass is: z + attention(z) ∘ (+1, -1), where z=(x,y).
• The attention(...) transform is learnable (using matrices P, Q).
• By training P, Q so that attention(z) ≈ eta*K*z (for some constant K), 
  the sign gate flips one coordinate, and adding this to z reproduces 
  the exact GDA step T(z).

The training objective is the mean squared error between the layer’s predicted update and the exact one-step GDA target for the second token:

L = (1 / 2B) * sum from b = 1 to B of || ẑ₁^(b) - z₁*^(b) ||₂²

This encourages P and Q to make Z + attention(Z) * diag(1, -1) match (x', y').

SAMPLE OUTPUT
------------
Step 0, Loss 2.07e-02
Step 200, Loss 6.84e-03
Step 400, Loss 7.02e-03
Step 600, Loss 6.38e-03
Step 800, Loss 6.59e-03
Final Test Loss: 6.76e-03
"""

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attention(P: torch.Tensor, Q: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Basic dot-product attention.
    Z: (B, 2, d), P and Q: (d, d).
    Returns an attention-based update for each token.
    """
    scores = torch.einsum('BNi,ij,BMj->BNM', Z, Q, Z)   # (B, 2, 2)
    keys = torch.einsum('ij,BNj->BNi', P, Z)           # (B, 2, d)

    # Strictly lower-triangular mask: token1 sees token0 only
    tril = torch.tensor([[0., 0.],
                         [1., 0.]], device=Z.device)

    out = torch.einsum('BNM,ML,BLi->BNi', scores, tril, keys)
    return out


class GDALayer(nn.Module):
    """
    Single layer that approximates one GDA step (x,y) -> (x*, y*).
    Implements: Z + attention(Z)*gate, where gate= (+1, -1).
    """
    def __init__(self, d=2, var=1e-3):
        super().__init__()
        self.P = nn.Parameter(torch.empty(d, d))
        self.Q = nn.Parameter(torch.empty(d, d))
        nn.init.normal_(self.P, 0.0, var)
        nn.init.normal_(self.Q, 0.0, var)

        # gate: +1 for x, -1 for y
        gate = torch.ones(d)
        gate[d // 2:] = -1
        self.register_buffer('gate', gate.view(1, 1, d))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (B,2,2). First token is context, second is query.
        Returns: updated Z with GDA-like step on token1.
        """
        return Z + attention(self.P, self.Q, Z) * self.gate


def batch(B=1024, eta=0.1):
    """
    Returns a batch of (Z_in, Z_tgt), each of shape (B,2,2).
    - Z_in: token0=(x,y), token1=(x,y)
    - Z_tgt: token0=(x,y), token1=(x',y') where (x', y') is one exact GDA step.
    """
    x, y = torch.randn(B, 1), torch.randn(B, 1)
    z_k = torch.cat([x, y], dim=1)

    # Gradients for A=1
    grad_x = x + y
    grad_y = -(x - y)
    z_next = torch.cat([x - eta * grad_x, y + eta * grad_y], dim=1)

    Z_in = torch.stack([z_k, z_k.clone()], dim=1)
    Z_tgt = torch.stack([z_k, z_next], dim=1)
    return Z_in.to(device), Z_tgt.to(device)


if __name__ == '__main__':
    torch.manual_seed(0)
    eta = 0.1
    model = GDALayer(d=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training loop
    for step in range(1000):
        Z_in, Z_tgt = batch(B=2048, eta=eta)
        loss = ((model(Z_in)[:, 1, :] - Z_tgt[:, 1, :]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f'Step {step}, Loss {loss.item():.2e}')

    # Test
    Z_in, Z_tgt = batch(B=4096, eta=eta)
    test_loss = ((model(Z_in)[:, 1, :] - Z_tgt[:, 1, :]) ** 2).mean().item()
    print(f'Final Test Loss: {test_loss:.2e}')
