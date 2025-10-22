"""
Implements the SAE training scheme from https://arxiv.org/abs/2406.04093.
Significant portions of this code have been copied from https://github.com/EleutherAI/sae/blob/main/sae
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


class ThresholdingAutoEncoderTopK(Dictionary, nn.Module):
    """
    Hard thresholding top-k autoencoder using a single decoder matrix.

    This implementation uses hard thresholding based on absolute values of activations,
    selecting the top-k features with largest absolute values while preserving their signs.
    Uses only a single weight matrix W_decoder for both encoding and decoding.

    Encoding: acts = (x - b_dec) @ W_decoder
    Selection: top-k based on |acts|, preserving signs
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(
        self, x: t.Tensor, return_topk: bool = False, use_threshold: bool = False
    ):
        # Hard thresholding: acts = (x - b_dec) @ W_decoder.T
        # Note: decoder.weight is stored as (activation_dim, dict_size) in nn.Linear
        # So we need to use decoder.weight without transpose
        feat_acts_BF = (x - self.b_dec) @ self.decoder.weight

        if use_threshold:
            # Apply threshold based on absolute values
            encoded_acts_BF = feat_acts_BF * (
                t.abs(feat_acts_BF) > self.threshold
            )
            if return_topk:
                # Get top-k based on absolute values
                abs_acts_BF = t.abs(feat_acts_BF)
                post_topk = abs_acts_BF.topk(self.k, sorted=False, dim=-1)
                top_indices_BK = post_topk.indices
                # Get the actual values (with signs) at the top-k indices
                top_acts_BK = feat_acts_BF.gather(-1, top_indices_BK)
                return (
                    encoded_acts_BF,
                    top_acts_BK,
                    top_indices_BK,
                    feat_acts_BF,
                )
            else:
                return encoded_acts_BF

        # Get top-k based on absolute values
        abs_acts_BF = t.abs(feat_acts_BF)
        post_topk = abs_acts_BF.topk(self.k, sorted=False, dim=-1)
        top_indices_BK = post_topk.indices

        # Get the actual values (with signs) at the top-k indices
        top_acts_BK = feat_acts_BF.gather(-1, top_indices_BK)

        buffer_BF = t.zeros_like(feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=top_acts_BK
        )

        if return_topk:
            return encoded_acts_BF, top_acts_BK, top_indices_BK, feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    def from_pretrained(path, k: Optional[int] = None, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)

        # Handle backward compatibility - check for encoder weights in old format
        if "encoder.weight" in state_dict:
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            # Remove encoder weights from state dict for new architecture
            del state_dict["encoder.weight"]
            if "encoder.bias" in state_dict:
                del state_dict["encoder.bias"]
        else:
            # New format - get dimensions from decoder
            activation_dim, dict_size = state_dict["decoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = ThresholdingAutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict, strict=False)  # Allow missing encoder keys
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class ThresholdingTopKTrainer(SAETrainer):
    """
    Hard thresholding Top-K SAE training scheme.

    Trains a sparse autoencoder using hard thresholding with a single decoder matrix,
    selecting features based on absolute activation values while preserving signs.
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class: type = ThresholdingAutoEncoderTopK,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,  # see Appendix A.2
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        k_anneal_steps: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "ThresholdingAutoEncoderTopK",
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
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.k_anneal_steps = k_anneal_steps

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialise autoencoder
        self.ae = dict_class(activation_dim, dict_size, k)
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
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

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None:
            return

        assert 0 <= k_anneal_steps < self.steps, (
            "k_anneal_steps must be >= 0 and < steps."
        )
        # self.k is the target k set for the trainer, not the dictionary's current k
        assert activation_dim > self.k, "activation_dim must be greater than k"

        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.k * ratio

        # Update in-place
        self.ae.k.fill_(int(annealed_value))

    def get_auxiliary_loss(self, residual_BD: t.Tensor, feat_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            # For hard thresholding, we work with absolute values for selection
            abs_acts_BF = t.abs(feat_acts_BF)
            auxk_latents = t.where(dead_features[None], abs_acts_BF, -t.inf)

            # Top-k dead latents based on absolute values
            auxk_abs_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Get the actual signed values at these indices
            auxk_acts = feat_acts_BF.gather(-1, auxk_indices)

            auxk_buffer_BF = t.zeros_like(feat_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, top_acts_BK: t.Tensor):
        device_type = "cuda" if top_acts_BK.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            # For hard thresholding, we work with absolute values
            active = t.abs(top_acts_BK).clone().detach()
            active[active == 0] = float("inf")  # Exclude exact zeros
            min_activations = active.min(dim=1).values.to(dtype=t.float32)
            min_activation = min_activations.mean()

            B, K = active.shape
            assert len(active.shape) == 2
            assert min_activations.shape == (B,)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x, step=None, logging=False):
        # Run the SAE
        f, top_acts_BK, top_indices_BK, feat_acts_BF = self.ae.encode(
            x, return_topk=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)

        x_hat = self.ae.decode(f)

        # Measure goodness of reconstruction
        e = x - x_hat

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts_BK.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = (
            self.get_auxiliary_loss(e.detach(), feat_acts_BF)
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
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        # Initialise the decoder bias
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "ThresholdingTopKTrainer",
            "dict_class": "ThresholdingAutoEncoderTopK",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }

