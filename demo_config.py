from dataclasses import dataclass, asdict
from typing import Optional, Type, Any
from enum import Enum
import torch as t

from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer, BatchTopKSAE
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"


@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype


@dataclass
class SparsityPenalties:
    standard: list[float]
    p_anneal: list[float]
    gated: list[float]


num_tokens = 50_000_000
eval_num_inputs = 1_000
random_seeds = [0]
expansion_factors = [8]

# note: learning rate is not used for topk
learning_rates = [3e-4]

wandb_project = "gemma-jumprelu_gated_sweep1"

LLM_CONFIG = {
    "EleutherAI/pythia-70m-deduped": LLMConfig(
        llm_batch_size=512, context_length=128, sae_batch_size=4096, dtype=t.float32
    ),
    "google/gemma-2-2b": LLMConfig(
        llm_batch_size=32, context_length=128, sae_batch_size=2048, dtype=t.bfloat16
    ),
}


# NOTE: In the current setup, the length of each sparsity penalty and target_l0 should be the same
SPARSITY_PENALTIES = {
    "EleutherAI/pythia-70m-deduped": SparsityPenalties(
        standard=[0.01, 0.05, 0.075, 0.1, 0.125, 0.15],
        p_anneal=[0.02, 0.03, 0.035, 0.04, 0.05, 0.075],
        gated=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
    ),
    "google/gemma-2-2b": SparsityPenalties(
        standard=[0.025, 0.035, 0.04, 0.05, 0.06, 0.07],
        p_anneal=[0.015, 0.025, 0.035, 0.04, 0.05, 0.06],
        gated=[0.02, 0.04, 0.05, 0.06, 0.07, 0.08],
    ),
}


TARGET_L0s = [20, 40, 80, 160, 320, 640]


@dataclass
class BaseTrainerConfig:
    activation_dim: int
    dict_size: int
    seed: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    lr: float
    l1_penalty: float
    warmup_steps: int = 1000
    resample_steps: Optional[int] = None


@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig):
    lr: float
    l1_penalty: float
    warmup_steps: int = 1000
    resample_steps: Optional[int] = None


@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig):
    lr: float
    initial_sparsity_penalty: float
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: Optional[int] = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10
    steps: Optional[int] = None


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    k: int
    auxk_alpha: float = 1 / 32
    decay_start: int = 24000
    threshold_beta: float = 0.999
    threshold_start_step: int = 10  # when to begin tracking the average threshold
    steps: Optional[int] = None


@dataclass
class GatedTrainerConfig(BaseTrainerConfig):
    lr: float
    l1_penalty: float
    warmup_steps: int = 1000
    resample_steps: Optional[int] = None


@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    lr: float
    target_l0: int
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001
    steps: Optional[int] = None


def get_trainer_configs(
    architectures: list[str],
    learning_rate: float,
    sparsity_index: int,
    seed: int,
    activation_dim: int,
    dict_size: int,
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
) -> list[dict]:
    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "dict_size": dict_size,
        "seed": seed,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }

    if TrainerType.P_ANNEAL.value in architectures:
        config = PAnnealTrainerConfig(
            **base_config,
            trainer=PAnnealTrainer,
            dict_class=AutoEncoder,
            lr=learning_rate,
            initial_sparsity_penalty=SPARSITY_PENALTIES[model_name].p_anneal[sparsity_index],
            steps=steps,
            wandb_name=f"PAnnealTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.STANDARD.value in architectures:
        config = StandardTrainerConfig(
            **base_config,
            trainer=StandardTrainer,
            dict_class=AutoEncoder,
            lr=learning_rate,
            l1_penalty=SPARSITY_PENALTIES[model_name].standard[sparsity_index],
            wandb_name=f"StandardTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.STANDARD_NEW.value in architectures:
        config = StandardNewTrainerConfig(
            **base_config,
            trainer=StandardTrainer,
            dict_class=AutoEncoderNew,
            lr=learning_rate,
            l1_penalty=SPARSITY_PENALTIES[model_name].standard[sparsity_index],
            wandb_name=f"StandardTrainerNew-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.TOP_K.value in architectures:
        config = TopKTrainerConfig(
            **base_config,
            trainer=TopKTrainer,
            dict_class=AutoEncoderTopK,
            k=TARGET_L0s[sparsity_index],
            steps=steps,
            wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K.value in architectures:
        config = TopKTrainerConfig(
            **base_config,
            trainer=BatchTopKTrainer,
            dict_class=BatchTopKSAE,
            k=TARGET_L0s[sparsity_index],
            steps=steps,
            wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.GATED.value in architectures:
        config = GatedTrainerConfig(
            **base_config,
            trainer=GatedSAETrainer,
            dict_class=GatedAutoEncoder,
            lr=learning_rate,
            l1_penalty=SPARSITY_PENALTIES[model_name].gated[sparsity_index],
            wandb_name=f"GatedTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.JUMP_RELU.value in architectures:
        config = JumpReluTrainerConfig(
            **base_config,
            trainer=JumpReluTrainer,
            dict_class=JumpReluAutoEncoder,
            lr=learning_rate,
            target_l0=TARGET_L0s[sparsity_index],
            wandb_name=f"JumpReluTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    return trainer_configs
