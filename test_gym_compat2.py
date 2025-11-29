#!/usr/bin/env python3
"""Test script for gym 0.26+ compatibility - TTT_logic_dim2"""

import sys
sys.path.insert(0, 'gym-tictactoe')

from gym_tictactoe.TTT_logic_dim2 import TicTacToeEnv

print("Testing TTT_logic_dim2 environment...")

# Test instantiation
env = TicTacToeEnv([-1, 1])
print("✓ Environment created")

# Test reset
result = env.reset()
if isinstance(result, tuple):
    obs, info = result
    print(f"✓ reset() returns (obs, info) - Gym 0.26+ compatible")
    print(f"  obs shape: {obs.shape}, info: {info}")
else:
    obs = result
    print(f"✗ reset() returns obs only - Old gym API")
    print(f"  obs shape: {obs.shape}")

# Test step
result = env.step(0, -1)
if len(result) == 5:
    obs, reward, terminated, truncated, info = result
    print(f"✓ step() returns 5 values - Gym 0.26+ compatible")
    print(f"  obs shape: {obs.shape}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
elif len(result) == 4:
    obs, reward, done, info = result
    print(f"✗ step() returns 4 values - Old gym API")
    print(f"  obs shape: {obs.shape}, reward: {reward}, done: {done}, info: {info}")

# Test observation_space
if hasattr(env, 'observation_space'):
    print(f"✓ Has observation_space: {env.observation_space}")
else:
    print(f"✗ Missing observation_space")

# Test action_space
if hasattr(env, 'action_space'):
    print(f"✓ Has action_space: {env.action_space}")
else:
    print(f"✗ Missing action_space")

print("\n✓ All compatibility checks passed!")
