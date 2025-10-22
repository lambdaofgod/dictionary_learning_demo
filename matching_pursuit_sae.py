"""
Implements the Matching Pursuit SAE training scheme.
Based on the structure of ThresholdingAutoEncoderTopK from https://arxiv.org/abs/2406.04093.
"""

import einops
import torch as t
import torch.nn as nn
from collections import namedtuple
from typing import Optional

from dictionary_learning.dictionary_learning.config import DEBUG
from dictionary_learning.dictionary_learning.dictionary import Dictionary
from dictionary_learning.dictionary_learning.trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


class MatchingPursuitAutoEncoder(Dictionary, nn.Module):
    """
    Matching Pursuit Autoencoder using a single decoder matrix.

    This implementation uses a greedy iterative algorithm that selects features
    one at a time based on maximum correlation with the residual, for S steps.
    Uses only a single weight matrix W_decoder for both encoding and decoding.

    Encoding: Iterative matching pursuit algorithm
    Selection: Feature with max absolute correlation at each step
    """

    def __init__(self, activation_dim: int, dict_size: int, s: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(s, int) and s > 0, f"s={s} must be a positive integer"
        self.register_buffer("S", t.tensor(s, dtype=t.int))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(
        self, x: t.Tensor, return_info: bool = False
    ):
        """
        Encode using matching pursuit algorithm.

        Args:
            x: Input activations of shape (batch, activation_dim)
            return_info: If True, return additional information for training

        Returns:
            If return_info=False: Sparse codes of shape (batch, dict_size)
            If return_info=True: (sparse_codes, selected_indices, initial_acts)
        """
        batch_size = x.shape[0]

        # Initialize residual: r = x - b_dec
        r = x - self.b_dec  # (batch, activation_dim)

        # Initialize sparse code
        z = t.zeros(
            batch_size, self.dict_size,
            device=x.device, dtype=x.dtype
        )

        # Track selected indices for each batch element at each step
        selected_indices = t.zeros(
            batch_size, self.S.item(),
            device=x.device, dtype=t.long
        )

        # Store initial activations (before iterative selection)
        initial_acts = (x - self.b_dec) @ self.decoder.weight

        # Matching pursuit iterations
        for step in range(self.S.item()):
            # Compute correlations: r @ W_decoder
            # Note: decoder.weight is stored as (activation_dim, dict_size) in nn.Linear
            correlations = r @ self.decoder.weight  # (batch, dict_size)

            # Find feature with maximum absolute correlation per position
            j = t.argmax(correlations.abs(), dim=-1)  # (batch,)
            selected_indices[:, step] = j

            # Extract correlation values for selected features
            # We need to gather along the last dimension
            j_expanded = j.unsqueeze(-1)  # (batch, 1)
            z_t = t.gather(correlations, -1, j_expanded).squeeze(-1)  # (batch,)

            # Update sparse code: accumulate selected features
            # We scatter_add to handle cases where same feature is selected multiple times
            z.scatter_add_(-1, j_expanded, z_t.unsqueeze(-1))

            # Update residual: r = r - z_t * W_dec[j]
            # Need to select the right dictionary elements for each position
            selected_dict = self.decoder.weight.T[j]  # (batch, activation_dim)
            r = r - z_t.unsqueeze(-1) * selected_dict

        if return_info:
            return z, selected_indices, initial_acts
        else:
            return z

    def decode(self, x: t.Tensor) -> t.Tensor:
        """Standard linear decode."""
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts = self.encode(x)
        x_hat = self.decode(encoded_acts)
        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts

    def scale_biases(self, scale: float):
        """Scale the decoder bias."""
        self.b_dec.data *= scale

    @classmethod
    def from_pretrained(cls, path, s: Optional[int] = None, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)

        # Get dimensions from decoder
        activation_dim, dict_size = state_dict["decoder.weight"].shape

        if s is None:
            s = state_dict["S"].item()
        elif "S" in state_dict and s != state_dict["S"].item():
            raise ValueError(f"s={s} != {state_dict['S'].item()}=state_dict['S']")

        autoencoder = MatchingPursuitAutoEncoder(activation_dim, dict_size, s)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class MatchingPursuitTrainer(SAETrainer):
    """
    Matching Pursuit SAE training scheme.

    Trains a sparse autoencoder using iterative matching pursuit algorithm,
    selecting features sequentially based on maximum correlation with residual.
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        s: int,  # number of matching pursuit steps
        layer: int,
        lm_name: str,
        dict_class: type = MatchingPursuitAutoEncoder,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,  # auxiliary loss coefficient
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        s_anneal_steps: Optional[int] = None,  # anneal S from activation_dim to s
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "MatchingPursuitAutoEncoder",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.s = s
        self.s_anneal_steps = s_anneal_steps

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialize autoencoder
        self.ae = dict_class(activation_dim, dict_size, s)
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic for auxiliary loss
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        # Optimizer and scheduler
        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def update_annealed_s(
        self, step: int, activation_dim: int, s_anneal_steps: Optional[int] = None
    ) -> None:
        """Update S buffer in-place with annealed value"""
        if s_anneal_steps is None:
            return

        assert 0 <= s_anneal_steps < self.steps, (
            "s_anneal_steps must be >= 0 and < steps."
        )
        # self.s is the target S set for the trainer, not the dictionary's current S
        assert activation_dim > self.s, "activation_dim must be greater than s"

        step = min(step, s_anneal_steps)
        ratio = step / s_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.s * ratio

        # Update in-place
        self.ae.S.fill_(int(annealed_value))

    def get_auxiliary_loss(self, residual: t.Tensor, initial_acts: t.Tensor):
        """
        Compute auxiliary loss for dead features.

        Args:
            residual: Reconstruction residual (x - x_hat)
            initial_acts: Initial activations before matching pursuit
        """
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            # For dead features, use initial activations (before MP selection)
            abs_acts = t.abs(initial_acts)
            auxk_latents = t.where(dead_features[None], abs_acts, -t.inf)

            # Top-k dead latents based on absolute values
            auxk_abs_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Get the actual signed values at these indices
            auxk_acts = initial_acts.gather(-1, auxk_indices)

            auxk_buffer = t.zeros_like(initial_acts)
            auxk_acts_full = auxk_buffer.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_full)
            l2_loss_aux = (
                (residual.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # Normalization similar to OpenAI implementation
            residual_mu = residual.mean(dim=0)[None, :].broadcast_to(
                residual.shape
            )
            loss_denom = (
                (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual.dtype, device=residual.device)

    def loss(self, x, step=None, logging=False):
        """Compute the loss for training."""
        # Run the SAE
        f, selected_indices, initial_acts = self.ae.encode(x, return_info=True)
        x_hat = self.ae.decode(f)

        # Measure goodness of reconstruction
        e = x - x_hat

        # Update the effective L0 (number of unique features selected)
        # For matching pursuit, we count unique features across S steps
        unique_features_per_sample = t.zeros(x.size(0), device=x.device)
        for i in range(x.size(0)):
            unique_features_per_sample[i] = selected_indices[i].unique().numel()
        self.effective_l0 = unique_features_per_sample.mean().item()

        # Update "number of tokens since fired" for each feature
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[selected_indices.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = (
            self.get_auxiliary_loss(e.detach(), initial_acts)
            if self.auxk_alpha > 0
            else 0
        )

        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item() if isinstance(auxk_loss, t.Tensor) else auxk_loss,
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        """Perform a training update step."""
        # Initialize the decoder bias
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        # Compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # Clip grad norm and remove grads parallel to decoder directions
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        # Do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_s(step, self.ae.activation_dim, self.s_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "MatchingPursuitTrainer",
            "dict_class": "MatchingPursuitAutoEncoder",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "s_anneal_steps": self.s_anneal_steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "s": self.ae.S.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }