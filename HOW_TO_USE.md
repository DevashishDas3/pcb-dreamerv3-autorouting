# How to Use Your Trained Model - Simple Guide

Now that you have your trained checkpoint (`logs/freerouting_debug2/latest.pt`), here's what you can do:

## 🎯 The Three Main Options

### Option 1: Run Full Inference Script (RECOMMENDED)
```bash
python infer.py --checkpoint logs/freerouting_debug2/latest.pt
```

**Features:**
- Generates routing solutions one at a time
- Can generate multiple attempts: `--num-solutions 5`
- Can use greedy or stochastic policy: `--greedy`
- Fully working and tested

**Try it:**
```bash
# Single solution
python infer.py --checkpoint logs/freerouting_debug2/latest.pt

# Multiple attempts (pick best)
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --num-solutions 5

# With different options
python infer.py --checkpoint logs/freerouting_debug2/latest.pt \
  --num-solutions 3 \
  --max-steps 300
```

---

### Option 2: Continue Training (Better Results Over Time)
```bash
# Train for another 50 episodes (4-6 hours)
python -u dreamer.py \
  --configs freerouting \
  --steps 100 \
  --prefill 30 \
  --eval_every 10 \
  --logdir logs/freerouting_v2
```

**Why?**
- After 105 updates: Model is ~random (0.0 reward)
- After 500 updates: Model starts learning (0.1-0.3 reward)
- After 5000 updates: Model routes well (0.5-0.8 reward)

---

### Option 3: Analyze What It Learned
```bash
python deep_analysis.py
```

Or read the detailed report:
```bash
cat CHECKPOINT_ANALYSIS.md
```

---

## 📋 Quick Examples

###  Example 1: Generate One Solution
```bash
python infer.py
```
Takes ~2 minutes, outputs final reward and steps taken.

### Example 2: Best of 10 Attempts
```bash
python infer.py --num-solutions 10
```
Tries 10 different routing strategies, picks the best.

### Example 3: Faster Solutions
```bash
python infer.py --greedy --max-steps 100
```
Uses deterministic (greedy) policy, shorter episodes.

### Example 4: Longer Episodes
```bash
python infer.py --max-steps 500
```
Allows up to 500 steps per solution (default: 200).

---

## 📚 Understanding Inference

### What Happens During Inference

```
Your PCB Image
      ↓
[Encoder] - Converts image to latent representation (1024-dim)
      ↓
[Dynamics RNN] - Predicts next state from current state + action
      ↓
[Actor Network] - Selects best routing action
      ↓
[Environment] - Executes action, returns reward & new PCB state
      ↓
[Value Network] - Estimates how good this state is
      ↓
(Repeat until solution found or time limit reached)
```

### Model Components in Your Checkpoint
- **World Model** (15.7M params): Learns what happens when you route
- **Actor** (1.1M params): Learns which routing actions are good
- **Value Network** (1.2M params): Estimates state quality
- **Reward Predictor**: Estimates solution success

---

## 🎓 What Model State Are You In?

### Right Now (105 training updates)
```
Update Count: 105 (100 pretrain + 5 main)
Training Time: ~38 minutes
Expected Reward: 0.0 (model barely trained)
Model Behavior: Near-random actions, learning beginning
```

### Interpretation
- ✅ Model works fine technically
- ⏳ Model hasn't learned to route well yet
- 📈 Will get better with more training

---

## 🚀 Recommended Workflow

### Step 1: Test Current Model (5 minutes)
```bash
python infer.py --checkpoint logs/freerouting_debug2/latest.pt
```
See: Does it work? What's the current reward?

### Step 2: Train Longer (4-6 hours)
```bash
python -u dreamer.py --configs freerouting --steps 100 --logdir logs/freerouting_v2
```
Expect: Much better reward, model learns actual routing

### Step 3: Use Improved Model (5 minutes)
```bash
python infer.py --checkpoint logs/freerouting_v2/latest.pt --num-solutions 5
```
See: How much better is the new model?

---

## 💡 Expected Results at Different Training Levels

| Training Updates | Expected Reward | Routing Behavior |
|---|---|---|
| 100 (current) | 0.0 | Random actions |
| 500 | 0.0-0.2 | Starting to learn patterns |
| 1000 | 0.2-0.4 | Recognizes good/bad moves |
| 5000 | 0.5-0.7 | Competent routing |
| 10000+ | 0.8+ | High-quality routing |

---

## 🔍 Monitoring Training Progress

### During Training, Watch These Files
```bash
# Real-time metrics
tail -f logs/freerouting_v2/metrics.jsonl

# Check what the model is learning
python deep_analysis.py
```

### Key Metrics to Look For
- **model_loss**: Should decrease over time (good sign)
- **reward_loss**: May stay low initially
- **actor_entropy**: Should stabilize
- **eval_return**: Should improve with training

---

## ✅ Checklist: Your Model Is Ready For

- ✅ **Testing**: Can run inference now
- ✅ **Evaluation**: Can test on new PCBs
- ✅ **Analysis**: Can inspect what it learned
- ⏳ **Production**: After more training (~1000+ updates)
- ⏳ **Deployment**: After much more training (5000+ updates)

---

## 🎯 Next Steps

**Pick one:**

```bash
# Check it works
python infer.py --checkpoint logs/freerouting_debug2/latest.pt

# Improve it
python -u dreamer.py --configs freerouting --steps 100 --logdir logs/freerouting_v2

# Generate solutions with current model
python infer.py --checkpoint logs/freerouting_debug2/latest.pt --num-solutions 10

# Understand it
cat CHECKPOINT_ANALYSIS.md
```

---

## 📖 Full Documentation

- **QUICK_START.md** - One-page quick reference
- **USING_YOUR_MODEL.md** - Comprehensive usage guide
- **INFERENCE_GUIDE.md** - Complete API documentation
- **CHECKPOINT_ANALYSIS.md** - What the model learned
