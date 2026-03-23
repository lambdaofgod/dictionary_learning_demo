import einops
import torch as t
import torch.nn as nn
from collections import namedtuple
from typing import Optional, Dict, Tuple, Union
from jaxtyping import Float, Int, Bool
import geoopt

from dictionary_learning.config import DEBUG
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)
from thresholding_sae import (
    NestedThresholdingAutoEncoderTopK,
    NestedThresholdingTopKTrainer,
    geometric_median,
)


class StiefelManifold(geoopt.manifolds.CanonicalStiefel):
    """
    Custom Stiefel manifold that handles dtype mismatches in transport operations.

    This is needed because geoopt's Stiefel manifold has issues with bfloat16
    in the _transp_follow_one method where torch.linalg.solve expects matching dtypes.

    WARNING: Stiefel manifold operations are VERY memory intensive!
    For dict_size=16384, activation_dim=768, each transport operation needs ~3 GB of memory!
    """

    @staticmethod
    def _amat(x: t.Tensor, u: t.Tensor) -> t.Tensor:
        """
        Compute the A matrix for Cayley transform.

        WARNING: This creates huge (dict_size, dict_size) matrices!
        Memory cost analysis for x, u both (dict_size, activation_dim):
        """
        # x and u are both (dict_size, activation_dim), e.g., (16384, 768)

        # x.transpose(-1, -2) is (activation_dim, dict_size), e.g., (768, 16384)
        # This is a VIEW - NO MEMORY ALLOCATED

        # u @ x.transpose(-1, -2) is (16384, 768) @ (768, 16384) = (16384, 16384)
        # ALLOCATES (dict_size, dict_size) matrix!
        # With dict_size=16384, float32: 16384^2 * 4 bytes = 1 GB!

        # x @ u.transpose(-1, -2) is another (16384, 768) @ (768, 16384) = (16384, 16384)
        # ALLOCATES ANOTHER 1 GB (dict_size, dict_size) matrix!

        # The subtraction creates a THIRD (16384, 16384) matrix
        # ALLOCATES ANOTHER 1 GB for the result!

        # TOTAL PEAK MEMORY: ~3 GB for dict_size=16384, float32
        return u @ x.transpose(-1, -2) - x @ u.transpose(-1, -2)

    def _transp_follow_one(
        self, x: t.Tensor, v: t.Tensor, *, u: t.Tensor
    ) -> t.Tensor:
        """
        Parallel transport along Cayley retraction.

        Memory cost analysis for x, u, v all (dict_size, activation_dim):
        """
        # x, u, v are all (dict_size, activation_dim), e.g., (16384, 768)

        # Calls _amat which creates (dict_size, dict_size) matrix
        # ALLOCATES 1 GB for result (the 3 GB during _amat computation is freed)
        a = self._amat(x, u)

        # a @ v is (16384, 16384) @ (16384, 768) = (16384, 768)
        # ALLOCATES dict_size * activation_dim * dtype_bytes (~50 MB, manageable)
        # v + result allocates another (16384, 768) for the sum (~50 MB)
        rhs = v + 1 / 2 * a @ v

        # -1/2 * a creates ANOTHER (16384, 16384) matrix
        # ALLOCATES ANOTHER 1 GB!
        lhs = -1 / 2 * a

        # In-place diagonal modification - no new allocation
        lhs[..., t.arange(a.shape[-2]), t.arange(x.shape[-2])] += 1

        # Ensure both lhs and rhs have the same dtype for linalg.solve
        # If either is bfloat16, convert both to float32 for the solve
        if lhs.dtype != rhs.dtype or lhs.dtype == t.bfloat16:
            # Store original dtype
            orig_dtype = lhs.dtype if lhs.dtype == t.bfloat16 else rhs.dtype

            # Convert to float32 for the solve
            lhs = lhs.to(t.float32)
            rhs = rhs.to(t.float32)

            # torch.linalg.solve needs LU decomposition of lhs (16384, 16384)
            # ALLOCATES ANOTHER 1+ GB internally for decomposition!
            # TOTAL MEMORY AT THIS POINT: ~3+ GB
            qv = t.linalg.solve(lhs, rhs)

            # Convert back to original dtype
            qv = qv.to(orig_dtype)
        else:
            # torch.linalg.solve needs LU decomposition of lhs (16384, 16384)
            # ALLOCATES ANOTHER 1+ GB internally for decomposition!
            qv = t.linalg.solve(lhs, rhs)

        return qv

    def proju(self, x: t.Tensor, u: t.Tensor) -> t.Tensor:
        """
        Project vector u onto tangent space at x.

        Also used as egrad2rgrad (Euclidean gradient -> Riemannian gradient).
        This is called EVERY optimization step!

        Memory cost analysis for x, u both (dict_size, activation_dim):
        """
        # x and u are both (dict_size, activation_dim), e.g., (16384, 768)

        # u.transpose(-1, -2) is (activation_dim, dict_size), e.g., (768, 16384)
        # This is a VIEW - NO MEMORY ALLOCATED

        # x @ u.transpose(-1, -2) is (16384, 768) @ (768, 16384) = (16384, 16384)
        # ALLOCATES (dict_size, dict_size) matrix!
        # With dict_size=16384, float32: 1 GB!

        # result @ x is (16384, 16384) @ (16384, 768) = (16384, 768)
        # ALLOCATES (dict_size, activation_dim) matrix (~50 MB, manageable)

        # u - result allocates another (dict_size, activation_dim) for subtraction
        # Another ~50 MB

        # TOTAL: ~1 GB for the intermediate (dict_size, dict_size) matrix
        # This happens EVERY gradient step!
        return u - x @ u.transpose(-1, -2) @ x

    def retr(self, x: t.Tensor, u: t.Tensor) -> t.Tensor:
        """
        Retraction: move from x in direction u on the manifold.

        Memory cost: Calls _transp_follow_one (see above for analysis).
        TOTAL: ~3+ GB per retraction step!
        """
        return self._transp_follow_one(x, x, u=u)

    def inner(
        self, x: t.Tensor, u: t.Tensor, v: t.Tensor = None, *, keepdim=False
    ) -> t.Tensor:
        """
        Riemannian inner product <u, v>_x on the tangent space at x.

        Memory cost analysis for x, u, v all (dict_size, activation_dim):
        """
        # x, u, v are all (dict_size, activation_dim), e.g., (16384, 768)

        # x.transpose(-1, -2) @ u is (768, 16384) @ (16384, 768) = (768, 768)
        # ALLOCATES (activation_dim, activation_dim) matrix (~2 MB, very manageable!)
        xtu = x.transpose(-1, -2) @ u

        if v is None:
            xtv = xtu
            v = u
        else:
            # x.transpose(-1, -2) @ v is another (768, 768) matrix (~2 MB)
            xtv = x.transpose(-1, -2) @ v

        # Element-wise operations and sums - minimal memory
        # u * v creates (dict_size, activation_dim) matrix (~50 MB)
        # xtv * xtu creates (activation_dim, activation_dim) matrix (~2 MB)
        # Sums reduce to scalar - no significant allocation

        # TOTAL: ~50 MB (very manageable compared to other operations!)
        return (u * v).sum([-1, -2], keepdim=keepdim) - 0.5 * (xtv * xtu).sum(
            [-1, -2], keepdim=keepdim
        )


class StiefelLinear(nn.Module):
    """
    A linear layer wrapper with weights constrained to the Stiefel manifold.

    This module wraps a ManifoldParameter on the Stiefel manifold, providing
    a drop-in replacement for nn.Linear(bias=False) where weights are kept
    orthonormal during optimization.

    Internal storage: (dict_size, activation_dim) to satisfy Stiefel constraint
    Exposed weight property: (activation_dim, dict_size) to match nn.Linear format
    """

    def __init__(self, dict_size: int, activation_dim: int):
        super().__init__()
        # Store weight as (dict_size, activation_dim) for Stiefel manifold
        # Stiefel requires shape[-1] <= shape[-2], so activation_dim <= dict_size
        assert activation_dim <= dict_size, (
            f"Stiefel manifold requires activation_dim <= dict_size, "
            f"got {activation_dim} > {dict_size}"
        )
        # Use our custom manifold that handles dtype mismatches
        manifold = StiefelManifold()
        # Initialize with proper Stiefel random matrix
        weight_data = manifold.random_naive(dict_size, activation_dim)
        self._weight = geoopt.ManifoldParameter(weight_data, manifold=manifold)

    @property
    def weight(self) -> t.Tensor:
        """
        Return weight transposed to match nn.Linear format: (activation_dim, dict_size)

        This allows the rest of the code to use self.decoder.weight as if it were
        a normal nn.Linear layer.
        """
        return self._weight.T

    def forward(
        self, x: Float[t.Tensor, "... dict_size"]
    ) -> Float[t.Tensor, "... activation_dim"]:
        """Forward pass: x @ _weight (internal storage, no transpose needed)"""
        return x @ self._weight


class StiefelNestedThresholdingAutoEncoderTopK(NestedThresholdingAutoEncoderTopK):
    """
    Nested hard thresholding top-k autoencoder with Stiefel manifold constraints.

    This implementation uses hard thresholding based on absolute values of activations,
    selecting different top-k values for nested models with varying sparsity levels.
    Uses a single weight matrix W_decoder constrained to the Stiefel manifold for both
    encoding and decoding.

    The Stiefel manifold constraint ensures the decoder weights remain orthonormal
    during training, replacing manual normalization with geometric optimization.

    Encoding: acts = (x - b_dec) @ W_decoder
    Selection: top-k based on |acts| for multiple k values, preserving signs
    """

    def __init__(self, activation_dim: int, dict_size: int, k_values: list[int]):
        # Validate and sort k values
        assert all(
            isinstance(k, int) and k > 0 for k in k_values
        ), "All k values must be positive integers"
        self.k_values = sorted(k_values)
        self.max_k = max(self.k_values)

        # Call grandparent (ThresholdingAutoEncoderTopK) init
        nn.Module.__init__(self)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        # Register k as buffer (from ThresholdingAutoEncoderTopK)
        self.register_buffer("k", t.tensor(self.max_k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        # Use StiefelLinear instead of nn.Linear - no normalization needed
        self.decoder = StiefelLinear(dict_size, activation_dim)

        # Decoder bias
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        # Store k_values as buffer for device handling
        self.register_buffer("k_tensor", t.tensor(self.k_values, dtype=t.int))

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        k_values: Optional[list[int]] = None,
        device: Optional[str] = None,
    ) -> "StiefelNestedThresholdingAutoEncoderTopK":
        """
        Load a pretrained Stiefel nested autoencoder from a file.
        """
        state_dict = t.load(path)

        # Get dimensions from decoder - try both regular and Stiefel keys
        decoder_weight_key = None
        for key in ["decoder._weight", "decoder.weight"]:
            if key in state_dict:
                decoder_weight_key = key
                break
        
        if decoder_weight_key is None:
            available_keys = [k for k in state_dict.keys() if 'decoder' in k]
            raise KeyError(
                f"Could not find decoder weight in state dict. "
                f"Available decoder keys: {available_keys}"
            )
        
        # For Stiefel, decoder._weight has shape (dict_size, activation_dim)
        # For regular, decoder.weight has shape (activation_dim, dict_size)
        if decoder_weight_key == "decoder._weight":
            dict_size, activation_dim = state_dict[decoder_weight_key].shape
        else:
            activation_dim, dict_size = state_dict[decoder_weight_key].shape

        if k_values is None:
            k_values = state_dict["k_tensor"].tolist()
        elif "k_tensor" in state_dict:
            saved_k = state_dict["k_tensor"].tolist()
            if sorted(k_values) != sorted(saved_k):
                raise ValueError(f"k_values={k_values} != saved k_values={saved_k}")

        autoencoder = cls(activation_dim, dict_size, k_values)
        
        # Use strict=False to allow missing keys from old checkpoints
        # and to handle different weight keys (decoder.weight vs decoder._weight)
        autoencoder.load_state_dict(state_dict, strict=False)
        
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class StiefelNestedThresholdingTopKTrainer(NestedThresholdingTopKTrainer):
    """
    Nested hard thresholding Top-K SAE training with Riemannian optimization.

    Trains a sparse autoencoder with multiple sparsity levels simultaneously,
    using hard thresholding with a single decoder matrix constrained to the
    Stiefel manifold. Uses RiemannianAdam optimizer to maintain the manifold
    constraint during training.
    """

    def __init__(
        self,
        steps: int,
        activation_dim: int,
        dict_size: int,
        k_values: list[int],
        layer: int,
        lm_name: str,
        k_weights: Optional[list[float]] = None,
        dict_class: type = StiefelNestedThresholdingAutoEncoderTopK,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "StiefelNestedThresholdingAutoEncoderTopK",
        submodule_name: Optional[str] = None,
    ):
        # Call parent init
        super().__init__(
            steps=steps,
            activation_dim=activation_dim,
            dict_size=dict_size,
            k_values=k_values,
            layer=layer,
            lm_name=lm_name,
            k_weights=k_weights,
            dict_class=dict_class,
            lr=lr,
            auxk_alpha=auxk_alpha,
            warmup_steps=warmup_steps,
            decay_start=decay_start,
            seed=seed,
            device=device,
            wandb_name=wandb_name,
            submodule_name=submodule_name,
        )

        # Replace optimizer with RiemannianAdam
        self.optimizer = geoopt.optim.RiemannianAdam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        # Re-create scheduler with new optimizer
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def update(self, step: int, x: Float[t.Tensor, "n_tokens activation_dim"]) -> float:
        """Perform a training update step with Riemannian optimization."""
        # Initialize the decoder bias
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        # Compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # Clip grad norm
        # Skip remove_gradient_parallel_to_decoder_directions - Riemannian optimizer handles tangent space
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        # Do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Skip set_decoder_norm_to_unit_norm - Stiefel manifold maintains constraint automatically

        return loss.item()
