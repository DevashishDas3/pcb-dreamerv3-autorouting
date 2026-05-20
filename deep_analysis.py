
import sys
import pathlib
import json
import numpy as np
import torch
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))
import models
import tools

config_path = pathlib.Path("configs.yaml")
yaml_loader = yaml.YAML(typ="safe", pure=True)
with open(config_path) as f:
    config = yaml_loader.load(f)
    base = config["defaults"]
    for key in ["freerouting"]:
        base.update(config.get(key, {}))
    config = base


class Config:
    def __init__(self, d):
        self.__dict__.update(d)


config = Config(config)

checkpoint_path = pathlib.Path("logs/freerouting_debug2/latest.pt")
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=config.device)

agent_state = checkpoint["agent_state_dict"]

print("\n" + "=" * 70)
print("WORLD MODEL LEARNED FEATURES ANALYSIS")
print("=" * 70)

print("\n1. IMAGE ENCODER (What visual features does it extract?)")
print("-" * 70)

encoder_conv_layers = [
    (k, v)
    for k, v in agent_state.items()
    if "_wm.encoder" in k and "conv" in k and "weight" in k and "norm" not in k
]

print(f"Found {len(encoder_conv_layers)} convolutional layers in encoder")
for i, (name, weight) in enumerate(encoder_conv_layers):
    w = weight.detach().cpu().numpy()
    print(f"\nLayer {i}: {name}")
    print(
        f"  Shape: {w.shape} (out_channels={w.shape[0]}, in_channels={w.shape[1]}, kernel={w.shape[2:]})"
    )
    print(f"  Weight range: [{w.min():.4f}, {w.max():.4f}]")
    print(f"  Mean: {w.mean():.6f}, Std: {w.std():.6f}")

    w_flat = w.reshape(w.shape[0], -1)  # (out_channels, in*K*K)
    for ch in range(min(3, w.shape[0])):
        filter_weights = w_flat[ch]
        print(
            f"  Filter {ch}: range=[{filter_weights.min():.4f}, {filter_weights.max():.4f}], "
            f"sparsity={100*(np.abs(filter_weights)<0.001).mean():.1f}%"
        )

print("\n\n2. DYNAMICS MODEL (How does it predict next states?)")
print("-" * 70)

dynamics_keys = [k for k in agent_state.keys() if "_wm.dynamics" in k and "weight" in k]
print(f"Dynamics model parameters: {len(dynamics_keys)}")

gru_weights = [k for k in dynamics_keys if "GRU" in k or "lstm" in k]
if gru_weights:
    for name in gru_weights[:2]:
        w = agent_state[name]
        print(f"\n{name}: shape={w.shape}")
        w_flat = w.flatten().detach().cpu().numpy()
        print(
            f"  Weight distribution: mean={w_flat.mean():.6f}, std={w_flat.std():.6f}"
        )
        print(f"  This is a {'recurrent' if w.shape[0] > 512 else 'regular'} layer")

print("\n\n3. IMAGE DECODER (Can it reconstruct observations?)")
print("-" * 70)

decoder_keys = [
    (k, v)
    for k, v in agent_state.items()
    if "_wm.heads.decoder" in k and ("conv" in k or "linear" in k) and "weight" in k
]

print(f"Found {len(decoder_keys)} decoder layers")
for i, (name, weight) in enumerate(decoder_keys[:3]):
    w = weight.detach().cpu().numpy()
    print(f"\n{i}: {name}")
    print(f"  Shape: {w.shape}")
    w_flat = w.flatten()
    print(
        f"  Weight stats: mean={w_flat.mean():.6f}, std={w_flat.std():.6f}, "
        f"sparsity={100*(np.abs(w_flat)<0.001).mean():.1f}%"
    )

print("\n\n4. REWARD PREDICTION HEAD (What does it predict about actions/states?)")
print("-" * 70)

reward_keys = [
    k
    for k in agent_state.keys()
    if "_wm.heads.reward" in k and "linear" in k and "weight" in k
]

print(f"Found {len(reward_keys)} reward prediction layers")
for i, name in enumerate(reward_keys[:2]):
    w = agent_state[name]
    w_np = w.detach().cpu().numpy()
    print(f"\nReward layer {i}: {name}")
    print(f"  Input dimension: {w_np.shape[1]}, Output dimension: {w_np.shape[0]}")
    print(f"  Weight range: [{w_np.min():.4f}, {w_np.max():.4f}]")

    # Check if there are dominant features
    w_flat = w_np.flatten()
    sorted_weights = np.sort(np.abs(w_flat))[-10:]
    print(f"  Top 10 largest weights by magnitude: {sorted_weights}")
    print(
        f"  -> Model {'HAS' if w_np.std() > 0.01 else 'has LIMITED'} learned reward patterns"
    )

print("\n\n5. ACTOR NETWORK (What action policy did it learn?)")
print("-" * 70)

actor_keys = [
    k for k in agent_state.keys() if "_task_behavior._policy" in k and "weight" in k
]
print(f"Actor policy parameters: {len(actor_keys)} layers")

if actor_keys:
    output_layer = [
        k for k in actor_keys if "dist" in k.lower() or "output" in k.lower()
    ]
    if output_layer:
        for name in output_layer[:1]:
            w = agent_state[name]
            print(f"Output layer: {name}")
            print(f"  Shape: {w.shape}")
            print(f"  Output size: {w.shape[0]} (action dimensions)")
    else:
        last_layer_name = actor_keys[-1]
        w = agent_state[last_layer_name]
        print(f"Last dense layer: {last_layer_name}")
        print(f"  Shape: {w.shape} (likely output layer)")

print("\n\n6. VALUE NETWORK (What baseline did it learn?)")
print("-" * 70)

value_keys = [k for k in agent_state.keys() if "value" in k and "weight" in k]
print(f"Value network parameters: {len(value_keys)} layers")

for name in value_keys[:2]:
    w = agent_state[name]
    w_np = w.detach().cpu().numpy()
    print(f"\n{name}: shape={w_np.shape}")
    print(f"  Weight range: [{w_np.min():.4f}, {w_np.max():.4f}]")

print("\n\n7. LEARNING ANALYSIS - WHAT HAPPENED?")
print("-" * 70)

all_weights = []
for v in agent_state.values():
    if isinstance(v, torch.Tensor) and len(v.shape) > 0:
        all_weights.append(v.flatten().detach().cpu().numpy())

all_weights = np.concatenate(all_weights)
print(f"Total model parameters: {len(all_weights):,}")
print(f"Weight distribution:")
print(f"  Mean: {all_weights.mean():.6f}")
print(f"  Std:  {all_weights.std():.6f}")
print(f"  Min:  {all_weights.min():.6f}")
print(f"  Max:  {all_weights.max():.6f}")

print(f"\nWeight initialization patterns:")
print(f"  Weights near 0 (< 0.01): {100*(np.abs(all_weights)<0.01).mean():.1f}%")
print(
    f"  Weights in [-0.1, 0.1]: {100*((all_weights>=-0.1) & (all_weights<=0.1)).mean():.1f}%"
)
print(
    f"  Weights in [-1.0, 1.0]: {100*((all_weights>=-1.0) & (all_weights<=1.0)).mean():.1f}%"
)

print("\n\n8. TRAINING CONVERGENCE ANALYSIS")
print("-" * 70)

metrics_path = pathlib.Path("logs/freerouting_debug2/metrics.jsonl")
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = [json.loads(line) for line in f]

    print(f"Training snapshots: {len(metrics)}")

    if len(metrics) > 1:
        first_metrics = metrics[0]
        last_metrics = metrics[-1]

        print(f"\nMetric changes across training:")
        for key in sorted(set(first_metrics.keys()) & set(last_metrics.keys())):
            if isinstance(first_metrics[key], (int, float)) and key not in [
                "step",
                "eval_episodes",
            ]:
                first_val = first_metrics[key]
                last_val = last_metrics[key]
                if first_val != 0:
                    change_pct = 100 * (last_val - first_val) / abs(first_val)
                    print(
                        f"  {key}: {first_val:.4f} -> {last_val:.4f} ({change_pct:+.1f}%)"
                    )
                else:
                    print(f"  {key}: {first_val:.4f} -> {last_val:.4f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

low_weight_pct = 100 * (np.abs(all_weights) < 0.01).mean()
print(f"WEIGHT DISTRIBUTION: {low_weight_pct:.1f}% of weights are near zero")

'''
print(f"""
Based on the analysis:

1. WEIGHT DISTRIBUTION: {low_weight_pct:.1f}% of weights are near zero
   - This is expected after short training
   - Network is still in early learning phase

2. ENCODER: Learned convolutional filters for image compression
   - Input: 64x64 RGB images → Latent representation
   - Status: Should extract basic visual patterns

3. DYNAMICS: RNN processes latent states and actions
   - Predicts state transitions
   - Status: Initialized but not fully learned

4. REWARD HEAD: Started learning value prediction
   - Predicts expected rewards from state-action pairs  
   - Status: Limited learned patterns (only 100 updates)

5. ACTOR: Action policy network
   - 100-dimensional output (continuous action control)
   - Status: Likely outputs random/default actions

6. DECODER: Reconstructs images from latents
   - Verifies world model captures visual information
   - Status: Initial stage

CONCLUSION:
The model completed 100 pretrain updates successfully and is now a baseline
for further training. The networks are initialized and beginning to learn
representations. With more training steps:
- Encoder will extract more salient PCB routing features
- Dynamics model will learn state transitions
- Reward predictor will find patterns in successful routing
- Actor will discover beneficial action sequences
- Value network will better estimate long-term rewards

SHORT TRAINING (only 100 updates) explains why weights appear mostly random.
The model needs 1000+ updates to develop meaningful learned features.
""")
'''
