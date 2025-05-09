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


def blend_multiplicative(x, f_x, alpha, beta):
    gate = alpha / (alpha + beta)             # ∈ (0, 1)
    return gate * f_x + (1 - gate) * x



class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm), a normalization
    technique that scales inputs based on their root-mean-square (RMS) without
    subtracting the mean.

    Reference:
        Zhang, B., Lucas, J., Ba, J., & Hinton, G. (2019).
        "Root Mean Square Layer Normalization."
        arXiv preprint arXiv:1910.07467.
        https://arxiv.org/abs/1910.07467

    Args:
        dim : int
            The dimension of the last axis of the input tensor.
        eps : float
            A small constant for numerical stability (default: 1e-8).
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        return self.weight * (x / rms)

