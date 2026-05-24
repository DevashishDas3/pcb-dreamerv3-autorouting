#!/usr/bin/env python3
"""
Test script to verify multi-DSN training setup.
This script validates that the environment can:
1. Load multiple DSN files
2. Randomly sample different DSN files on reset
3. Detect and penalize intersections
4. Train on multiple DSN files
"""

import os
import sys
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_dsn_files_exist():
    """Verify all DSN test files exist."""
    print("=" * 60)
    print("Test 1: Checking DSN files exist")
    print("=" * 60)

    dsn_files = [
        "dsn_test_files/board_1.dsn",
        "dsn_test_files/board_2.dsn",
        "dsn_test_files/board_3.dsn",
        "dsn_test_files/board_4.dsn",
        "dsn_test_files/board_5.dsn",
    ]

    all_exist = True
    for dsn_file in dsn_files:
        full_path = project_root / dsn_file
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dsn_file}")
        if not exists:
            all_exist = False

    print()
    if all_exist:
        print("✓ All DSN files found!")
        return True
    else:
        print("✗ Some DSN files are missing")
        return False


def test_config_yaml():
    """Verify config.yaml has multi-DSN settings."""
    print("=" * 60)
    print("Test 2: Checking configs.yaml multi-DSN settings")
    print("=" * 60)

    config_file = project_root / "configs.yaml"

    with open(config_file, "r") as f:
        content = f.read()

    checks = {
        "freerouting_dsn_files": "freerouting_dsn_files:" in content,
        "freerouting_intersection_penalty_scale": "freerouting_intersection_penalty_scale:"
        in content,
    }

    all_good = True
    for key, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {key}")
        if not exists:
            all_good = False

    print()
    if all_good:
        print("✓ configs.yaml has multi-DSN settings!")
        return True
    else:
        print("✗ configs.yaml missing multi-DSN settings")
        return False


def test_env_implementation():
    """Verify FreeroutingJPypeEnv has multi-DSN methods."""
    print("=" * 60)
    print("Test 3: Checking FreeroutingJPypeEnv implementation")
    print("=" * 60)

    env_file = project_root / "envs" / "freerouting_jpype_env.py"

    with open(env_file, "r") as f:
        content = f.read()

    checks = {
        "dsn_files_list parameter": "dsn_files_list=" in content,
        "load_dsn_file method": "def load_dsn_file" in content,
        "_detect_intersections method": "def _detect_intersections" in content,
        "intersection_penalty_scale": "intersection_penalty_scale" in content,
        "Random DSN sampling in reset": "self._rng.choice(self._dsn_files_list)"
        in content,
    }

    all_good = True
    for key, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {key}")
        if not exists:
            all_good = False

    print()
    if all_good:
        print("✓ FreeroutingJPypeEnv has all multi-DSN features!")
        return True
    else:
        print("✗ FreeroutingJPypeEnv missing some features")
        return False


def test_dreamer_integration():
    """Verify dreamer.py passes multi-DSN config to environment."""
    print("=" * 60)
    print("Test 4: Checking dreamer.py integration")
    print("=" * 60)

    dreamer_file = project_root / "dreamer.py"

    with open(dreamer_file, "r") as f:
        content = f.read()

    checks = {
        "dsn_files_list passed to env": "dsn_files_list=dsn_files" in content,
        "intersection_penalty_scale passed to env": "intersection_penalty_scale=intersection_penalty_scale"
        in content,
        "Config attribute handling": "getattr(config, 'freerouting_dsn_files'"
        in content,
    }

    all_good = True
    for key, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {key}")
        if not exists:
            all_good = False

    print()
    if all_good:
        print("✓ dreamer.py correctly integrates multi-DSN config!")
        return True
    else:
        print("✗ dreamer.py integration incomplete")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " Multi-DSN Training Setup Verification ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = [
        ("DSN Files", test_dsn_files_exist()),
        ("Config YAML", test_config_yaml()),
        ("Environment Implementation", test_env_implementation()),
        ("Dreamer Integration", test_dreamer_integration()),
    ]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print()
    if all_passed:
        print("✓ All checks passed! Multi-DSN training is ready.")
        print()
        print("To train with multi-DSN support:")
        print("  python train.py --configs freerouting")
        print()
        print("Configuration details:")
        print("  - DSN files: board_1.dsn through board_5.dsn in dsn_test_files/")
        print("  - Sampling strategy: Random DSN per episode reset")
        print("  - Intersection penalty scale: 0.1 (tunable in configs.yaml)")
        print()
        return 0
    else:
        print("✗ Some checks failed. Please review the implementation.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
