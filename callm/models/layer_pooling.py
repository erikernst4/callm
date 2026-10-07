import torch
import torch.nn as nn
import numpy as np


def create_layer_weights(
    num_layers: int,
    init_strategy: str = "uniform",
    init_std: float | None = None,
    init_k: float = 3.0,
) -> torch.Tensor:
    """
    Creates an initial raw weight tensor of shape (num_layers,) for layer pooling modules.

    Strategies:
      - 'uniform' / 'zeros': All zeros -> softmax yields uniform weights (1 / num_layers).
      - 'normal' / 'gaussian': Gaussian centered at middle layer (mu = (num_layers - 1) / 2).
          std defaults to num_layers / 6.0 unless specified via init_std.
          Raw weights set to -0.5 * ((i - mu) / std)^2 so softmax matches Gaussian PDF.
      - 'reverse_chi' / 'chi': Chi distribution peaking at middle layer with k degrees of freedom.
          First layer receives near-zero weight, middle receives peak, late layers receive decaying tail.
          Raw weights set to (k - 1) * log(x + eps) - 0.5 * x^2.
      - 'random' / 'rand': Standard Gaussian random values N(0, 1).
    """
    strategy = init_strategy.lower()
    if strategy in ("uniform", "zeros"):
        return torch.zeros(num_layers)

    if strategy in ("normal", "gaussian"):
        if num_layers <= 1:
            return torch.zeros(num_layers)
        mu = (num_layers - 1) / 2.0
        std = init_std if init_std is not None else (num_layers / 6.0)
        idx = torch.arange(num_layers, dtype=torch.float32)
        log_pdf = -0.5 * ((idx - mu) / std) ** 2
        return log_pdf

    if strategy in ("reverse_chi", "chi"):
        if num_layers <= 1:
            return torch.zeros(num_layers)
        k = init_k
        peak_x = float(np.sqrt(max(k - 1.0, 1e-3)))
        mid_idx = (num_layers - 1) / 2.0
        idx = torch.arange(num_layers, dtype=torch.float32)
        x = peak_x * (idx / mid_idx)
        eps = 1e-7
        log_pdf = (k - 1.0) * torch.log(x + eps) - 0.5 * (x**2)
        return log_pdf

    if strategy in ("random", "rand"):
        if num_layers <= 1:
            return torch.zeros(num_layers)
        std = init_std if init_std is not None else 1.0
        return torch.randn(num_layers) * std

    raise ValueError(
        f"Unknown layer weight init_strategy: '{init_strategy}'. "
        f"Expected one of ['uniform', 'zeros', 'normal', 'gaussian', 'reverse_chi', 'chi', 'random', 'rand']."
    )


class LayerWeightedAverage(nn.Module):
    """
    Learns a weight for each layer and computes the weighted average of
    activations across the layer dimension.
    """

    def __init__(
        self,
        num_layers: int,
        init_strategy: str = "uniform",
        init_std: float | None = None,
        init_k: float = 3.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.init_strategy = init_strategy
        self.init_std = init_std
        self.init_k = init_k
        self.layer_weights = nn.Parameter(
            create_layer_weights(num_layers, init_strategy, init_std, init_k)
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, num_layers, #seq, #model_dim) or (#batch, num_layers, #model_dim)
        Returns:
            weighted_activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim)
        """
        weights = torch.softmax(self.layer_weights, dim=0).to(dtype=activations.dtype)

        if activations.ndim == 4:
            weights = weights.view(1, self.num_layers, 1, 1)
        elif activations.ndim == 3:
            weights = weights.view(1, self.num_layers, 1)
        else:
            raise ValueError(
                f"LayerWeightedAverage expected 3D or 4D input, got {activations.ndim}D"
            )

        return (activations * weights).sum(dim=1)

    def get_weights(self) -> torch.Tensor:
        """Returns the normalized (softmax) layer weights."""
        return torch.softmax(self.layer_weights, dim=0)


class GlobalZNormWeightedAverage(nn.Module):
    """
    Global Z-normalization followed by learned weighted average across layers.

    Normalizes all layer activations using a single global (mean, std) computed
    over all layers and all samples, then applies softmax-weighted average.
    This removes gross scale differences but uses the same statistics for every
    layer.

    The normalization statistics must be injected via
    ``set_normalization_stats()`` before the first forward pass.
    """

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        init_strategy: str = "uniform",
        init_std: float | None = None,
        init_k: float = 3.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.init_strategy = init_strategy
        self.init_std = init_std
        self.init_k = init_k
        self.layer_weights = nn.Parameter(
            create_layer_weights(num_layers, init_strategy, init_std, init_k)
        )
        # Buffers for normalization stats (injected later)
        self.register_buffer("global_mean", torch.zeros(model_dim))
        self.register_buffer("global_std", torch.ones(model_dim))
        self._stats_set = False

    def set_normalization_stats(
        self, global_mean: torch.Tensor, global_std: torch.Tensor
    ):
        """Inject precomputed global normalization statistics."""
        self.global_mean.copy_(global_mean)
        self.global_std.copy_(global_std.clamp(min=1e-8))
        self._stats_set = True

    def load_stats(self, layer_stats: dict):
        """Set normalization statistics from a datamodule's ``layer_stats``."""
        self.set_normalization_stats(
            layer_stats["global_mean"].to(self.global_mean.device),
            layer_stats["global_std"].to(self.global_std.device),
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, num_layers, #seq, #model_dim) or (#batch, num_layers, #model_dim)
        Returns:
            weighted_activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim)
        """
        # Normalize: (x - mean) / std
        activations = (activations - self.global_mean) / self.global_std

        weights = torch.softmax(self.layer_weights, dim=0).to(dtype=activations.dtype)

        if activations.ndim == 4:
            weights = weights.view(1, self.num_layers, 1, 1)
        elif activations.ndim == 3:
            weights = weights.view(1, self.num_layers, 1)
        else:
            raise ValueError(
                f"GlobalZNormWeightedAverage expected 3D or 4D input, got {activations.ndim}D"
            )

        return (activations * weights).sum(dim=1)

    def get_weights(self) -> torch.Tensor:
        """Returns the normalized (softmax) layer weights."""
        return torch.softmax(self.layer_weights, dim=0)


class LayerwiseZNormWeightedAverage(nn.Module):
    """
    Layer-wise Z-normalization followed by learned weighted average across layers.

    Each layer's activations are normalized using that layer's own (mean, std)
    computed over all samples. This puts all layers on the same statistical
    footing so the softmax weights only need to learn *importance*, not fight
    scale differences.

    The normalization statistics must be injected via
    ``set_normalization_stats()`` before the first forward pass.
    """

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        init_strategy: str = "uniform",
        init_std: float | None = None,
        init_k: float = 3.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.init_strategy = init_strategy
        self.init_std = init_std
        self.init_k = init_k
        self.layer_weights = nn.Parameter(
            create_layer_weights(num_layers, init_strategy, init_std, init_k)
        )
        # Buffers for per-layer normalization stats (injected later)
        self.register_buffer("layer_means", torch.zeros(num_layers, model_dim))
        self.register_buffer("layer_stds", torch.ones(num_layers, model_dim))
        self._stats_set = False

    def set_normalization_stats(
        self, layer_means: torch.Tensor, layer_stds: torch.Tensor
    ):
        """
        Inject precomputed per-layer normalization statistics.

        Args:
            layer_means: shape (num_layers, model_dim)
            layer_stds: shape (num_layers, model_dim)
        """
        self.layer_means.copy_(layer_means)
        self.layer_stds.copy_(layer_stds.clamp(min=1e-8))
        self._stats_set = True

    def load_stats(self, layer_stats: dict):
        """Set normalization statistics from a datamodule's ``layer_stats``."""
        self.set_normalization_stats(
            layer_stats["layer_means"].to(self.layer_means.device),
            layer_stats["layer_stds"].to(self.layer_stds.device),
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, num_layers, #seq, #model_dim) or (#batch, num_layers, #model_dim)
        Returns:
            weighted_activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim)
        """
        # Per-layer normalization
        if activations.ndim == 4:
            # (num_layers, model_dim) -> (1, num_layers, 1, model_dim)
            means = self.layer_means.unsqueeze(0).unsqueeze(2)
            stds = self.layer_stds.unsqueeze(0).unsqueeze(2)
        elif activations.ndim == 3:
            # (num_layers, model_dim) -> (1, num_layers, model_dim)
            means = self.layer_means.unsqueeze(0)
            stds = self.layer_stds.unsqueeze(0)
        else:
            raise ValueError(
                f"LayerwiseZNormWeightedAverage expected 3D or 4D input, got {activations.ndim}D"
            )

        activations = (activations - means) / stds

        weights = torch.softmax(self.layer_weights, dim=0).to(dtype=activations.dtype)

        if activations.ndim == 4:
            weights = weights.view(1, self.num_layers, 1, 1)
        elif activations.ndim == 3:
            weights = weights.view(1, self.num_layers, 1)

        return (activations * weights).sum(dim=1)

    def get_weights(self) -> torch.Tensor:
        """Returns the normalized (softmax) layer weights."""
        return torch.softmax(self.layer_weights, dim=0)


class LayerNormWeightedAverage(nn.Module):
    """
    Learnable LayerNorm per layer followed by learned weighted average.

    Applies an independent ``nn.LayerNorm(model_dim)`` to each layer's
    activations before computing the softmax-weighted average. This can
    capture per-layer affine transformations.

    Note: the LLM's own transformer blocks already contain LayerNorm, so
    the benefit of an additional learnable LayerNorm is uncertain. This
    also introduces extra learnable parameters that are jointly optimized
    with the weighted average — the same issue as the baseline.
    """

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        init_strategy: str = "uniform",
        init_std: float | None = None,
        init_k: float = 3.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.init_strategy = init_strategy
        self.init_std = init_std
        self.init_k = init_k
        self.layer_weights = nn.Parameter(
            create_layer_weights(num_layers, init_strategy, init_std, init_k)
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(num_layers)]
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, num_layers, #seq, #model_dim) or (#batch, num_layers, #model_dim)
        Returns:
            weighted_activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim)
        """
        # Apply per-layer LayerNorm
        normed_layers = []
        for i in range(self.num_layers):
            layer_act = activations[:, i]  # (#batch, #seq, #dim) or (#batch, #dim)
            normed_layers.append(self.layer_norms[i](layer_act))
        activations = torch.stack(normed_layers, dim=1)

        weights = torch.softmax(self.layer_weights, dim=0).to(dtype=activations.dtype)

        if activations.ndim == 4:
            weights = weights.view(1, self.num_layers, 1, 1)
        elif activations.ndim == 3:
            weights = weights.view(1, self.num_layers, 1)
        else:
            raise ValueError(
                f"LayerNormWeightedAverage expected 3D or 4D input, got {activations.ndim}D"
            )

        return (activations * weights).sum(dim=1)

    def get_weights(self) -> torch.Tensor:
        """Returns the normalized (softmax) layer weights."""
        return torch.softmax(self.layer_weights, dim=0)


class SingleLayerSelector(nn.Module):
    """
    Selects activations from a single specified layer index.
    """

    def __init__(self, layer_idx: int, num_layers: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = num_layers

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, num_layers, #seq, #model_dim) or (#batch, num_layers, #model_dim)
        Returns:
            layer_activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim)
        """
        if activations.ndim not in (3, 4):
            raise ValueError(
                f"SingleLayerSelector expected 3D or 4D input, got {activations.ndim}D"
            )

        n_layers = activations.shape[1]
        resolved_idx = self.layer_idx % n_layers
        return activations[:, resolved_idx]
