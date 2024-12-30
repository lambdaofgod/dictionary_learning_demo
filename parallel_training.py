#!/usr/bin/env python3
import subprocess
import time
import os

# Configuration for 4x 3090s
# For our current implementation, relative training speed is:
# standard / p_anneal > top_k > batch_top_k > jump_relu > gated
# So, we have jump_relu and gated on their own GPUs
configurations = [
    {
        "arch": "jump_relu",
        "layers": 12,
        "device": "cuda:0"
    },
    {
        "arch": "top_k p_anneal",
        "layers": 12,
        "device": "cuda:1"
    },
    {
        "arch": "batch_top_k standard",
        "layers": 12,
        "device": "cuda:2"
    },
    {
        "arch": "gated",
        "layers": 12,
        "device": "cuda:3"
    },
]

SAVE_DIR = "trained_saes"
MODEL_NAME = "google/gemma-2-2b"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Launch jobs
for i, config in enumerate(configurations):
    log_file = f"logs/{(config['arch'].replace(' ', '_'))}_l{config['layers']}_{config['device'].replace(':', '_')}.out"
    
    cmd = [
        "python", "demo.py",
        "--save_dir", SAVE_DIR,
        "--model_name", MODEL_NAME,
        "--architectures", config["arch"],
        "--layers", str(config["layers"]),
        "--device", config["device"],
        "--use_wandb"
    ]
    
    # Launch with nohup
    with open(log_file, "w") as f:
        subprocess.Popen(
            f"nohup {' '.join(cmd)} > {log_file} 2>&1",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    print(f"Started job {i + 1}/{len(configurations)}: {config['arch']} with layers: {config['layers']}")
    time.sleep(2)

print("All jobs submitted!")