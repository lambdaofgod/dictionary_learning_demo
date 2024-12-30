#!/usr/bin/env python3
import subprocess
import time
import os

# Configuration for 3x 3090s
configurations = [
    {
        "arch": ("standard", "gated"),
        "layers": 12,
        "device": "cuda:0"
    },
    {
        "arch": ("top_k", "batch_top_k"),
        "layers": 12,
        "device": "cuda:1"
    },
    {
        "arch": ("jump_relu", "p_anneal"),
        "layers": 12,
        "device": "cuda:2"
    }
]

SAVE_DIR = "trained_saes"
MODEL_NAME = "google/gemma-2-2b"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Launch jobs
for i, config in enumerate(configurations):
    arch_pair = " ".join(config["arch"])
    log_file = f"logs/{'_'.join(config['arch'])}_l{config['layers']}_{config['device'].replace(':', '_')}.out"
    
    cmd = [
        "python", "demo.py",
        "--save_dir", SAVE_DIR,
        "--model_name", MODEL_NAME,
        "--architectures", arch_pair,
        "--layers", str(config["layers"]),
        "--device", config["device"]
    ]
    
    # Launch with nohup
    with open(log_file, "w") as f:
        subprocess.Popen(
            f"nohup {' '.join(cmd)} > {log_file} 2>&1",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    print(f"Started job {i}/3: {arch_pair} with {config['layers']} layers")
    time.sleep(2)

print("All jobs submitted!")