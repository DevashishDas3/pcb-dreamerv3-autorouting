#!/usr/bin/env python3
"""
Analyze what the DreamerV3 world model learned from the checkpoint.
"""

import sys
import pathlib
import json
import numpy as np
import torch
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))
import models
import tools

# Load config
config_path = pathlib.Path("configs.yaml")
yaml_loader = yaml.YAML(typ="safe", pure=True)
with open(config_path) as f:
    config = yaml_loader.load(f)
    base = config["defaults"]
    for key in ["freerouting"]:
        base.update(config.get(key, {}))
    config = base


# Convert dict to namespace-like object
class Config:
    def __init__(self, d):
        self.__dict__.update(d)


config = Config(config)

# Load checkpoint
checkpoint_path = pathlib.Path("logs/freerouting_debug2/latest.pt")
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=config.device)

print("\n" + "=" * 60)
print("CHECKPOINT STRUCTURE")
print("=" * 60)
print(f"Keys in checkpoint: {list(checkpoint.keys())}")

print("\n" + "=" * 60)
print("AGENT STATE DICT ANALYSIS")
print("=" * 60)

agent_state = checkpoint["agent_state_dict"]
print(f"Total number of parameters: {len(agent_state)}")

# Group by module
modules = {}
for key in agent_state.keys():
    module_name = key.split(".")[0]
    if module_name not in modules:
        modules[module_name] = []
    modules[module_name].append(key)

print(f"\nModules found: {list(modules.keys())}")

# Analyze each module
for module_name in sorted(modules.keys()):
    keys = modules[module_name]
    total_params = sum(agent_state[k].numel() for k in keys)
    print(f"\n{module_name}:")
    print(f"  - Parameters: {len(keys)} tensors")
    print(f"  - Total elements: {total_params:,}")

    # Show first few keys and their shapes
    for key in keys[:3]:
        tensor = agent_state[key]
        print(f"    - {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    if len(keys) > 3:
        print(f"    ... and {len(keys)-3} more")

print("\n" + "=" * 60)
print("WORLD MODEL ENCODER ANALYSIS")
print("=" * 60)

# Extract encoder weights
encoder_keys = [k for k in agent_state.keys() if "_wm" in k and "encoder" in k]
print(f"Encoder parameters: {len(encoder_keys)}")

# Try to reconstruct and analyze learned image encoder
if encoder_keys:
    # Show some encoder layer information
    conv_layers = [k for k in encoder_keys if "conv" in k and "weight" in k]
    print(f"Convolutional layers: {len(conv_layers)}")
    for i, key in enumerate(conv_layers[:3]):
        w = agent_state[key]
        print(
            f"  Conv layer {i}: {w.shape} (in_channels={w.shape[1]}, out_channels={w.shape[0]}, kernel={w.shape[2:]})"
        )
        # Analyze weight statistics
        w_flat = w.flatten()
        print(
            f"    - Weight stats: mean={w_flat.mean():.4f}, std={w_flat.std():.4f}, "
            f"min={w_flat.min():.4f}, max={w_flat.max():.4f}"
        )

print("\n" + "=" * 60)
print("WORLD MODEL DECODER ANALYSIS")
print("=" * 60)

decoder_keys = [k for k in agent_state.keys() if "_wm" in k and "decoder" in k]
print(f"Decoder parameters: {len(decoder_keys)}")

if decoder_keys:
    # Show some decoder layer information
    conv_layers = [k for k in decoder_keys if "conv" in k and "weight" in k]
    print(f"Convolutional layers: {len(conv_layers)}")
    for i, key in enumerate(conv_layers[:3]):
        w = agent_state[key]
        print(f"  Conv layer {i}: {w.shape}")

print("\n" + "=" * 60)
print("DYNAMICS MODEL (RNN) ANALYSIS")
print("=" * 60)

# Look for RNN/LSTM parameters
rnn_keys = [
    k
    for k in agent_state.keys()
    if "_wm" in k and ("rnn" in k or "gru" in k or "lstm" in k)
]
print(f"RNN parameters: {len(rnn_keys)} tensors")
if rnn_keys:
    for key in rnn_keys[:5]:
        tensor = agent_state[key]
        print(f"  - {key}: shape={tensor.shape}")

print("\n" + "=" * 60)
print("ACTOR/VALUE NETWORK ANALYSIS")
print("=" * 60)

actor_keys = [k for k in agent_state.keys() if "task_behavior" in k or "actor" in k]
value_keys = [k for k in agent_state.keys() if "value" in k]

print(f"Actor network parameters: {len(actor_keys)} tensors")
print(f"Value network parameters: {len(value_keys)} tensors")

# Analyze actor layers
if actor_keys:
    dense_layers = [
        k for k in actor_keys if "dense" in k or "linear" in k or "weight" in k
    ]
    print(f"  Dense layers in actor: {len(dense_layers)}")
    for key in dense_layers[:3]:
        w = agent_state[key]
        print(f"    - {key}: shape={w.shape}")

print("\n" + "=" * 60)
print("LEARNED FEATURES INSPECTION")
print("=" * 60)

# Check if there are any learned embeddings with meaningful patterns
print("\nAnalyzing weight distributions across key layers:")

sample_keys = [
    k
    for k in agent_state.keys()
    if ("conv" in k or "dense" in k or "linear" in k) and "weight" in k
][:5]

for key in sample_keys:
    w = agent_state[key]
    w_flat = w.flatten().detach().cpu().numpy()
    print(f"\n{key}:")
    print(f"  Shape: {w.shape}")
    print(f"  Mean: {w_flat.mean():.6f}")
    print(f"  Std:  {w_flat.std():.6f}")
    print(f"  Min:  {w_flat.min():.6f}")
    print(f"  Max:  {w_flat.max():.6f}")
    print(f"  Sparsity (% near zero): {100 * (np.abs(w_flat) < 0.01).mean():.1f}%")

# Load metrics
print("\n" + "=" * 60)
print("TRAINING METRICS")
print("=" * 60)

metrics_path = pathlib.Path("logs/freerouting_debug2/metrics.jsonl")
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = [json.loads(line) for line in f]

    print(f"Total metrics snapshots: {len(metrics)}")
    if metrics:
        last_metrics = metrics[-1]
        print("\nLatest training metrics:")
        for key in sorted(last_metrics.keys())[:10]:
            value = last_metrics[key]
            if isinstance(value, (int, float)):
                print(
                    f"  {key}: {value:.4f}"
                    if isinstance(value, float)
                    else f"  {key}: {value}"
                )

print("\n" + "=" * 60)
print("SUMMARY INSIGHTS")
print("=" * 60)

print("""
The checkpoint contains:
1. Encoder: Learns image representations from 64x64 RGB observations
2. Dynamics Model (RNN): Predicts next latent states from current state + action
3. Decoder: Reconstructs images from latent states (image reconstruction loss)
4. Other heads: Reward prediction, terminal/continuation prediction
5. Actor: Learns action policy from latent states
6. Value: Learns state value estimates for planning

Key learning indicators:
- Image encoder/decoder: Check if weight distributions show meaningful filters
- Dynamics model: Should learn temporal correlations in the PCB routing environment
- Actor: Should learn action distribution for controlling router
- Weights near zero indicate possible dead neurons or underfitting
""")
