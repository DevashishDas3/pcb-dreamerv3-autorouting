# Using Your Trained Model for PCB Routing Inference

Now that you have a trained DreamerV3 checkpoint, here's everything you can do with it:

## Quick Start: Generate Routing Solutions

```bash
# Single solution attempt
python infer.py --checkpoint logs/freerouting_debug2/latest.pt

# Multiple attempts (pick the best)
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --num-solutions 5

# Use deterministic policy (no randomness)
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --greedy

# Longer episodes
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --max-steps 500
```

---

## How Inference Works

### 1. **Load Checkpoint**
Your checkpoint contains:
- **Encoder**: Converts PCB images → latent representations
- **Dynamics Model (GRU)**: Predicts next states
- **Actor Network**: Outputs routing actions
- **Value Network**: Estimates state quality
- **Reward Head**: Predicts routing success

### 2. **Processing Pipeline**

```
PCB Image (64×64×3)
     ↓
[Encoder] → Latent Representation (1024-dim)
     ↓
[Dynamics GRU] → Updated Latent State
     ↓
[Actor Network] → Action Distribution → Sample Action
     ↓
[Environment Step] → New PCB State, Reward
     ↓
[Repeat] → Generate whole trajectory
```

### 3. **What Each Component Does**

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Encoder** | PCB image | Latent vector | Compress visual information |
| **Dynamics** | (latent, action) | Next latent | Predict state transitions |
| **Actor** | Features | Action distribution | Generate routing actions |
| **Value** | Features | Q-value | Estimate route quality |
| **Reward** | Features | Predicted reward | Estimate routing success |

---

## Inference Strategies

### Strategy 1: Single Greedy Solution (Fast)
```bash
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --greedy
```
- **Speed**: Fast (deterministic, no sampling)
- **Quality**: May be suboptimal (follows mode, no exploration)
- **When to use**: Quick sampling, benchmarking

### Strategy 2: Multiple Stochastic Attempts (Best Quality)
```bash
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --num-solutions 5
```
- **Speed**: Slower (5 full rollouts)
- **Quality**: High (can explore different solutions)
- **When to use**: Production routing, high-quality solutions needed

### Strategy 3: Extended Rollout (Longer Routes)
```bash
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --max-steps 1000
```
- Allows longer episodes (default: 200 steps)
- Useful for complex PCBs needing more steps

---

## What Makes a Good Routing Solution?

The model evaluates solutions on:

1. **Reward Signal** (main metric)
   - Higher = better routing
   - Currently near 0.0 (model just starting to learn)

2. **Episode Length**
   - Fewer steps = faster routing
   - The model learns to route efficiently

3. **Success Rate**
   - Did it reach a goal state?
   - Tracked in trajectory

---

## Practical Workflows

### Workflow 1: Test Current Model (Sanity Check)
```bash
# Quick test - does the model work on new problems?
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --greedy
```
**Expected output:**
- PCB image processes without errors
- Actions sampled from policy
- Reward score (likely 0.0 since model just trained)

### Workflow 2: Collect Multiple Solutions
```bash
# Attempt 10 different routings, save the best
python infer.py --checkpoint logs/freerouting_debug2/latest.pt \
  --num-solutions 10 \
  --max-steps 300
```
**Benefits:**
- Explore different routing strategies
- Select highest-reward solution
- Estimate model's stochasticity/variance

### Workflow 3: Benchmark Over Many Problems
```bash
# Process different PCB files, measure performance
for pcb in pcbs/*.dsn; do
  python infer.py --checkpoint logs/freerouting_debug2/latest.pt \
    --num-solutions 3 \
    --max-steps 200
done
```

---

## What Happens During Inference

### Simple Case: Single Solution
1. **Initialize**: Reset environment, get initial PCB observation
2. **Encode Loop** (200 steps):
   - Pass PCB image through encoder → 1024-dim latent
   - Update RNN state with action
   - Sample action from actor policy
   - Execute action in environment
   - Accumulate reward
3. **Return**: Best achieved reward and trajectory

### Advanced Case: Multiple Attempts + Selection
1. Run simple case 5 times with different random seeds
2. Compare final rewards
3. Return best trajectory
4. Also return comparison statistics (mean, std, min, max rewards)

---

## Expected Performance (Current Checkpoint)

### Model Status
- **Updates Trained**: 105 (100 pretrain + 5)
- **Data Collected**: ~4,000 environment steps
- **Expected Performance**: Near-random

### Realistic Expectations
```
Model State          → Expected Reward
✅ Just initialized     0.0 (baseline)
⏳ Training 100 updates 0.0-0.1 (no learning yet)
✅ Training 1000 updates 0.1-0.5 (learning begins)
🚀 Training 10K updates  0.5+ (meaningful routing)
🎯 Training 100K updates 0.8+ (good routing)
```

**Your current model**: Likely outputs near-random actions. As you train longer, it will discover better routing patterns.

---

## Extending the Inference Script

### Add Custom Metrics
```python
def analyze_solution(solution):
    """Extract routing statistics"""
    rewards = [t['reward'] for t in solution['trajectory']]
    actions = np.array([t['action'] for t in solution['trajectory']])
    
    return {
        'total_reward': sum(rewards),
        'avg_step_reward': np.mean(rewards),
        'action_variance': np.std(actions),
        'num_steps': len(solution['trajectory']),
    }
```

### Save Solutions to Disk
```python
import json

solution = agent.generate_solution(env)
with open('routing_solution.json', 'w') as f:
    json.dump({
        'reward': float(solution['total_reward']),
        'steps': solution['num_steps'],
        'actions': solution['trajectory'],  # Actions taken
    }, f, indent=2)
```

### Visualize Trajectory
```python
def plot_rewards(solution):
    import matplotlib.pyplot as plt
    rewards = [t['reward'] for t in solution['trajectory']]
    plt.plot(rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f"Trajectory (Total: {sum(rewards):.2f})")
    plt.savefig('trajectory.png')
```

---

## Comparison: Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Goal** | Learn routing policy | Use learned policy |
| **Mode** | Agent learns (requires gradients) | Agent is frozen (eval mode) |
| **Randomness** | Samples actions (explores) | Can be greedy (deterministic) |
| **Updates** | Model weights change | Model weights fixed |
| **Speed** | ~22s per update | ~few seconds per solution |
| **Data** | Learns from collected episodes | Uses pre-learned weights |

---

## Troubleshooting

### Problem: "Checkpoint not found"
```bash
# Make sure you trained first
python -u dreamer.py --configs freerouting --steps=20 --prefill=10 \
  --logdir=logs/freerouting_debug2
```

### Problem: "CUDA out of memory" during inference
```bash
# Use CPU instead
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --device cpu
```

### Problem: Model outputs all zeros/NaNs
- Indicates checkpoint not loaded correctly
- Check file path is correct
- Verify checkpoint file is valid: `python -c "import torch; torch.load('logs/freerouting_debug2/latest.pt')"`

### Problem: Reward always 0.0
- **Expected!** With only 100 training updates, model hasn't learned routing
- Train longer: `--steps 100 --prefill 20`
- Reward signal will improve as model learns

---

## Next Steps

### 1. **Short-term**: Test current model
```bash
python infer.py --checkpoint logs/freerouting_debug2/latest.pt \
  --num-solutions 5
```
Compare rewards, check that inference works correctly.

### 2. **Medium-term**: Train longer
```bash
# 2-4 hour training run
python -u dreamer.py \
  --configs freerouting \
  --steps 100 \
  --prefill 20 \
  --eval_every 10 \
  --logdir logs/freerouting_v2
```
Expect 500-600 updates, better routing performance.

### 3. **Long-term**: Production deployment
```bash
# After successful training
python infer.py \
  --checkpoint logs/freerouting_v2/latest.pt \
  --num-solutions 10 \
  --max-steps 500
```
Generate high-quality routes by sampling multiple attempts.

---

## API Reference: InferenceAgent

```python
from infer import InferenceAgent
import torch

# Load agent
agent = InferenceAgent(
    checkpoint_path='logs/freerouting_debug2/latest.pt',
    config=config,
    device='cuda'  # or 'cpu'
)

# Single solution
result = agent.generate_solution(
    env,
    num_steps=200,
    greedy=True,  # deterministic or stochastic
    verbose=True
)
print(f"Reward: {result['total_reward']:.4f}")
print(f"Steps: {result['num_steps']}")

# Multiple attempts
results = agent.solve_with_multiple_attempts(
    env,
    num_attempts=5,
    num_steps=200,
    verbose=True
)
best_solution = results['best_solution']
best_reward = results['best_reward']
```

---

## Summary

**You now have:**
1. ✅ Trained checkpoint with learned world model
2. ✅ Inference script to generate routing solutions
3. ✅ Ability to test on new PCB problems
4. ✅ Framework for comparing multiple solution attempts

**Next:** Run `python infer.py` to generate your first routing solutions!
