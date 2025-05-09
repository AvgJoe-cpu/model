class ScalarGate(nn.Module):
    """
    Implements the bounded, symmetric, exponential gating mechanism:

        • g : ℝᵈ → ℝ   (learnable scalar – here a linear layer)
        • s(x) = c · tanh(g(x))
        • α(x) = exp( s(x) )
        • β(x) = exp(−s(x))   (⇒ α · β = 1)

    The module **only** returns the scalars (α, β); how you blend them
    with x and f(x) is up to the calling code.
    """
    def __init__(self, in_dim: int, c: float = 2.0):
        """
        Args
        ----
        in_dim : size of the feature vector x (g’s input dimension)
        c      : constant range bound (s ∈ [−c, c]  ⇒  α,β ∈ [e^(−c), e^c])
        """
        super().__init__()
        self.g = nn.Linear(in_dim, 1)   # learnable scalar function g(x)
        self.c = c

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        α : torch.Tensor, shape (batch, 1, …)   -- α(x) = exp(s(x))
        β : torch.Tensor, shape (batch, 1, …)   -- β(x) = exp(−s(x))
        s : torch.Tensor, shape (batch, 1, …)   -- s(x) itself for inspection
        """
        # s(x) = c · tanh(g(x))     ← Equation (1)
        s = self.c * torch.tanh(self.g(x))

        # α(x) = exp(s(x)),  β(x) = exp(−s(x))   ← Equation (2)
        alpha = torch.exp(s)
        beta  = torch.exp(-s)        # reciprocal by construction

        return alpha, beta, s


def gated_residual(x: torch.Tensor,
                   f_x: torch.Tensor,
                   alpha: torch.Tensor,
                   beta:  torch.Tensor,
                   normalise: bool = False):
    """
    Combines the skip path x and transform path f(x) with α, β.

        • raw   : out = α·x + β·f(x)
        • norm. : out = (α·x + β·f(x)) / (α + β)

    Shapes
    ------
    x, f_x  : (..., d)           (must be broadcast-compatible)
    alpha,
    beta    : (..., 1) or (...)  (broadcast over the feature dimension)
    """
    out = alpha * x + beta * f_x           # ← Equation (3) numerator
    if normalise:
        out = out / (alpha + beta)         # ← Equation (3) denominator
    return out

