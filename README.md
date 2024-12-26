Demonstration of how to train, evaluate, and plot evaluation results using the `dictionary_learning` repo. We use `dictionary_learning` as a git submodule.

# Setup

```
git submodule update --init --recursive
pip install -e .
```

# Usage

In `demo_config.py`, we set various hyperparameters like number of training tokens and expansion factors. We also set various SAE specific hyperparameters, such as the bandwidth for JumpReLU.

The bottom of `demo.py` contains a variety of example commands for training a variety of SAEs on different models. As an example, the following command will traing 30 different SAEs (6 sparsities per architecture) on 50 million tokens on Pythia-70M in ~1 hour on a RTX 3090. It will also evaluate all SAEs on metrics like L0 and loss recovered.

`python demo.py --save_dir ./saes --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures standard jump_relu batch_top_k top_k gated`

We currently support the following SAEs:

- ReLU
- JumpReLU
- TopK
- BatchTopK
- Gated
- P-anneal ReLU

NOTE: For TopK and BatchTopK, we record the average minimum activation value during training, enabling the use of the SAEs using a simple JumpReLU threshold during inference. We use this as the default approach during inference, as it tends to get a better loss recovered, eliminates interaction between features, and enables encoding only a subset of latents.

We can then graph the results using `graphing.ipynb` if we specify the names of the output folders.