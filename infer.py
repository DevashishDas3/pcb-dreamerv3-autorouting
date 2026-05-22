#!/usr/bin/env python3
"""
Inference Script: Use trained DreamerV3 model to solve PCB routing problems

This script demonstrates how to:
1. Load a trained checkpoint
2. Feed it new PCB routing problems
3. Generate routing solutions
4. Compare multiple solution attempts
"""

import sys
import pathlib
import argparse
import numpy as np
import torch
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import models
from dreamer import Dreamer
import tools
import exploration as expl
import envs.wrappers as wrappers


class InferenceAgent:
    """Wrapper for running inference with a trained DreamerV3 model"""

    def __init__(self, checkpoint_path, config, obs_space, act_space, device="cpu"):
        self.device = device
        self.config = config
        self.checkpoint_path = checkpoint_path

        class DummyLogger:
            step = 0

            def scalar(self, *args, **kwargs):
                pass

            def write(self, *args, **kwargs):
                pass

            def video(self, *args, **kwargs):
                pass

        class DummyDataset:
            def __next__(self):
                raise StopIteration

        # Build the same agent class used for training, but without a dataset.
        self.agent = Dreamer(
            obs_space, act_space, config, DummyLogger(), DummyDataset()
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.agent.to(device)
        self.agent.eval()

        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        print(f"   Device: {device}")
        print(
            f"   Model parameters: {sum(p.numel() for p in self.agent.parameters()):,}"
        )

    def generate_solution(self, env, num_steps=None, greedy=True, verbose=True):
        """
        Generate a routing solution by rolling out the trained policy

        Args:
            env: Freerouting environment
            num_steps: Max steps (default: env.spec.max_episode_steps)
            greedy: Use greedy policy (mode) instead of sampling
            verbose: Print progress

        Returns:
            dict with solution data and trajectory
        """
        if num_steps is None:
            num_steps = self.config.time_limit

        def pack_obs(obs, is_first, is_terminal, reward=0.0, discount=1.0):
            packed = {k: np.asarray(v)[None] for k, v in obs.items()}
            packed["is_first"] = np.array([is_first], dtype=np.float32)
            packed["is_terminal"] = np.array([is_terminal], dtype=np.float32)
            packed["reward"] = np.array([reward], dtype=np.float32)
            packed["discount"] = np.array([discount], dtype=np.float32)
            return packed

        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs, info = reset_result, {}
        obs = pack_obs(obs, True, False)
        state = None
        trajectory = []
        total_reward = 0.0
        terminated = False
        truncated = False

        if verbose:
            print(f"\n🤖 Generating routing solution ({num_steps} steps max)...")

        for step in range(num_steps):
            # Convert observation to tensor
            obs_torch = self.agent._wm.preprocess(obs)

            with torch.no_grad():
                # Forward pass through world model
                embed = self.agent._wm.encoder(obs_torch)
                action_prev = None if state is None else state[1]

                latent, _ = self.agent._wm.dynamics.obs_step(
                    state[0] if state else None,
                    action_prev,
                    embed,
                    obs_torch["is_first"],
                )

                # Get features and sample action
                feat = self.agent._wm.dynamics.get_feat(latent)

                if greedy:
                    # Deterministic policy (no exploration)
                    actor = self.agent._task_behavior.actor(feat)
                    policy_action = actor.mode()
                else:
                    # Stochastic policy (with sampling)
                    actor = self.agent._task_behavior.actor(feat)
                    policy_action = actor.sample()

                # Update state
                state = (latent, policy_action.detach())
                action = policy_action[0].detach().cpu().numpy()

            # Step environment
            step_result = env.step({"action": action})
            if isinstance(step_result, tuple) and len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                terminated = bool(done)
                truncated = False
            obs = pack_obs(
                obs,
                False,
                terminated or truncated,
                reward=reward,
                discount=info.get("discount", 1.0),
            )
            trajectory.append(
                {"step": step, "action": action.copy(), "reward": reward, "info": info}
            )
            total_reward += reward

            if verbose and (step + 1) % max(1, num_steps // 10) == 0:
                print(f"  Step {step+1}/{num_steps}, Reward so far: {total_reward:.4f}")

            if terminated or truncated:
                if verbose:
                    print(f"  ✓ Episode ended at step {step+1}")
                break

        result = {
            "success": terminated and not truncated,
            "total_reward": total_reward,
            "num_steps": len(trajectory),
            "trajectory": trajectory,
            "final_obs": obs,
            "env_info": info,
        }

        if verbose:
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Steps taken: {len(trajectory)}")

        return result

    def solve_with_multiple_attempts(
        self, env, num_attempts=3, num_steps=None, verbose=True
    ):
        """
        Try multiple solution attempts and return the best one

        Args:
            env: Freerouting environment
            num_attempts: Number of different solutions to try
            num_steps: Max steps per attempt
            verbose: Print progress

        Returns:
            dict with best solution and comparison stats
        """
        best_solution = None
        best_reward = -np.inf
        all_attempts = []

        if verbose:
            print(f"\n📊 Running {num_attempts} solution attempts...")

        for attempt in range(num_attempts):
            if verbose:
                print(f"\n--- Attempt {attempt + 1}/{num_attempts} ---")

            solution = self.generate_solution(
                env, num_steps=num_steps, greedy=False, verbose=verbose
            )
            all_attempts.append(solution)

            if solution["total_reward"] > best_reward:
                best_reward = solution["total_reward"]
                best_solution = solution

            # Reset env for next attempt
            env.reset()

        if verbose:
            print(f"\n🏆 Best solution:")
            print(f"   Reward: {best_solution['total_reward']:.4f}")
            print(f"   Steps: {best_solution['num_steps']}")
            print(f"\n📈 All attempts:")
            for i, sol in enumerate(all_attempts, 1):
                print(
                    f"   Attempt {i}: reward={sol['total_reward']:.4f}, steps={sol['num_steps']}"
                )

        return {
            "best_solution": best_solution,
            "all_attempts": all_attempts,
            "best_reward": best_reward,
        }


def create_env(config):
    """Create a freerouting environment for inference"""
    import gymnasium as gym
    from envs import freerouting_jpype_env

    # Create the freerouting environment
    print("🔧 Creating Freerouting environment...")
    env = freerouting_jpype_env.FreeroutingJPypeEnv(
        jar_path=config.freerouting_jar_path,
        dsn_file_path=config.freerouting_dsn_path,
        seed=config.seed,
    )

    # Wrap it
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    return env


def main():
    parser = argparse.ArgumentParser(description="DreamerV3 PCB Routing Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/freerouting_debug2/latest.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--num-solutions",
        type=int,
        default=1,
        help="Number of solution attempts to generate",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Max steps per solution"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy (deterministic) policy instead of sampling",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")

    args = parser.parse_args()

    # Load config
    config_path = pathlib.Path("configs.yaml")
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    with open(config_path) as f:
        config_dict = yaml_loader.load(f)
        base = config_dict["defaults"]

        def recursive_update(target, update):
            for key, value in update.items():
                if (
                    isinstance(value, dict)
                    and key in target
                    and isinstance(target[key], dict)
                ):
                    recursive_update(target[key], value)
                else:
                    target[key] = value

        recursive_update(base, config_dict.get("freerouting", {}))
        config_dict = base

    class Config:
        def __init__(self, d):
            self.__dict__.update(d)

    config = Config(config_dict)
    config.device = args.device

    # Create environment first so we can use the real observation/action spaces.
    env = create_env(config)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # Load agent
    checkpoint_path = pathlib.Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Run training first: python dreamer.py --configs freerouting --steps=50")
        return

    agent = InferenceAgent(
        checkpoint_path,
        config,
        env.observation_space,
        env.action_space,
        device=args.device,
    )

    # Generate solutions
    print("\n" + "=" * 70)
    print("GENERATING PCB ROUTING SOLUTIONS")
    print("=" * 70)

    if args.num_solutions == 1:
        solution = agent.generate_solution(
            env, num_steps=args.max_steps, greedy=args.greedy, verbose=True
        )
        print(f"\n✅ Solution generated!")
        print(f"   Final reward: {solution['total_reward']:.4f}")
        print(f"   Steps taken: {solution['num_steps']}")
    else:
        results = agent.solve_with_multiple_attempts(
            env, num_attempts=args.num_solutions, num_steps=args.max_steps, verbose=True
        )
        print(f"\n✅ All solutions generated!")

    env.close()

    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print("""
Generate a single solution:
  python infer.py --checkpoint logs/freerouting_debug2/latest.pt

Generate 5 solutions and pick the best:
  python infer.py --checkpoint logs/freerouting_debug2/latest.pt --num-solutions 5

Use greedy (deterministic) policy:
  python infer.py --checkpoint logs/freerouting_debug2/latest.pt --greedy

Use GPU if available:
  python infer.py --checkpoint logs/freerouting_debug2/latest.pt --device cuda

Increase max steps:
  python infer.py --checkpoint logs/freerouting_debug2/latest.pt --max-steps 500
""")


if __name__ == "__main__":
    main()
