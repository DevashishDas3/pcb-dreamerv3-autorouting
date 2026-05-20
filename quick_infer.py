#!/usr/bin/env python3
"""
Quick Example: Load checkpoint and generate one routing solution
Simplified version that loads components directly.
"""

import sys
import pathlib
import numpy as np
import torch
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))
import models
import tools
import envs.wrappers as wrappers
from envs import freerouting_jpype_env

def load_config():
    """Load configuration from configs.yaml"""
    config_path = pathlib.Path("configs.yaml")
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    with open(config_path) as f:
        config_dict = yaml_loader.load(f)
        base = config_dict["defaults"]
        base.update(config_dict.get("freerouting", {}))
    
    class Config:
        def __init__(self, d):
            self.__dict__.update(d)
    return Config(base)

def load_agent(checkpoint_path, config, device='cpu'):
    """Load agent components from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent_state = checkpoint['agent_state_dict']
    
    # Create a dummy environment to get the proper space objects
    obs_space = {'image': np.zeros((64, 64, 3), dtype=np.uint8)}
    act_space = np.zeros((100,), dtype=np.float32)
    
    # Create proper gymnasium spaces
    import gymnasium
    obs_space_dict = gymnasium.spaces.Dict({
        'image': gymnasium.spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
    })
    act_space_obj = gymnasium.spaces.Box(-1.0, 1.0, shape=(100,), dtype=np.float32)
    
    wm = models.WorldModel(obs_space_dict, act_space_obj, 0, config)
    task_behavior = models.ImagBehavior(config, wm)
    
    # Load weights - handle the nested structure
    wm_state = {k.replace('_wm.', ''): v for k, v in agent_state.items() if k.startswith('_wm.')}
    task_behavior_state = {k.replace('_task_behavior.', ''): v for k, v in agent_state.items() if k.startswith('_task_behavior.')}
    
    wm.load_state_dict(wm_state, strict=False)
    task_behavior.load_state_dict(task_behavior_state, strict=False)
    
    wm.to(device)
    task_behavior.to(device)
    wm.eval()
    task_behavior.eval()
    
    return wm, task_behavior

def main():
    print("\n" + "="*70)
    print("DreamerV3 PCB Routing Inference - Quick Example")
    print("="*70)
    
    # 1. Load configuration
    print("\n1️⃣  Loading configuration...")
    config = load_config()
    device = 'cpu'  # Use CPU (change to 'cuda' if available)
    
    # 2. Load the trained model
    print("2️⃣  Loading trained checkpoint...")
    checkpoint_path = pathlib.Path("logs/freerouting_debug2/latest.pt")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("\n📝 To create a checkpoint, run training first:")
        print("   python -u dreamer.py --configs freerouting --steps=20 --prefill=10")
        return
    
    # Load world model and behavior networks
    print("   Loading model components...")
    wm, task_behavior = load_agent(checkpoint_path, config, device=device)
    
    print(f"✅ Checkpoint loaded!")
    total_params = sum(p.numel() for p in wm.parameters()) + sum(p.numel() for p in task_behavior.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # 3. Create environment
    print("\n3️⃣  Setting up PCB routing environment...")
    env = freerouting_jpype_env.FreeroutingJPypeEnv(
        jar_path=config.freerouting_jar_path,
        dsn_file_path=config.freerouting_dsn_path,
        seed=42,
    )
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    
    print("✅ Environment ready!")
    
    # 4. Generate routing solution
    print("\n4️⃣  Generating routing solution...")
    print(f"   Max steps: {config.time_limit}")
    
    obs, info = env.reset()
    state = None
    total_reward = 0.0
    step_count = 0
    
    for step in range(config.time_limit):
        with torch.no_grad():
            # Preprocess observation
            obs_torch = wm.preprocess(obs)
            
            # Encode observation
            embed = wm.encoder(obs_torch)
            
            # Update world model state
            action_prev = None if state is None else state[1]
            latent, _ = wm.dynamics.obs_step(
                state[0] if state else None,
                action_prev,
                embed,
                obs_torch["is_first"]
            )
            
            # Get action from actor (greedy/deterministic)
            feat = wm.dynamics.get_feat(latent)
            actor = task_behavior.actor(feat)
            action = actor.mode().detach().cpu().numpy()
            
            # Update state
            state = (latent, torch.tensor(action, device=device, dtype=torch.float32))
        
        # Execute action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Progress indicator
        if (step + 1) % max(1, config.time_limit // 5) == 0:
            print(f"   Step {step+1}/{config.time_limit}, Reward: {total_reward:.4f}")
        
        if terminated or truncated:
            break
    
    # 5. Report results
    print("\n5️⃣  Routing Complete!")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"   Total Reward: {total_reward:.4f}")
    print(f"   Steps Taken: {step_count}/{config.time_limit}")
    print(f"   Status: {'✅ Success' if terminated else '⏸️  Truncated'}")
    print(f"\n💡 Interpretation:")
    if total_reward > 0.0:
        print(f"   ✅ Model found a rewarding path!")
    else:
        print(f"   ℹ️  Model reward is {total_reward:.4f}")
        print(f"      (Expected for early training: model still learning)")
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   - Train longer: python -u dreamer.py --configs freerouting --steps=100")
    print("   - Multiple solutions: python infer.py --num-solutions 5")
    print("   - Full guide: see INFERENCE_GUIDE.md")
    print("="*70 + "\n")
    
    env.close()

if __name__ == "__main__":
    main()
