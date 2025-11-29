#!/usr/bin/env python3
"""Test config 0 (Q-table) with gym 0.26+ compatibility"""

import sys
sys.path.insert(0, './gym-tictactoe')

# Import the environment
from gym_tictactoe.TTT_plain import TicTacToeEnv
env = TicTacToeEnv(symbols=[-1, 1], board_size=3, win_size=3)

# Import the Q-table agent
from RL_Qtable import Qtable

# Initialize Q-table
RL = Qtable(
    action_dim=env.action_space.n,
    state_dim=env.state_space.shape[0],
    learning_rate=0.8,
    gamma=0.9,
)

print("✓ Config 0 initialization successful")
print(f"  action_dim: {env.action_space.n}")
print(f"  state_dim: {env.state_space.shape[0]}")

# Test a few game steps
state, info = env.reset()
print(f"✓ reset() works: state shape = {state.shape}")

# AI move
action1 = RL.choose_action(state)
print(f"✓ choose_action() works: action = {action1}")

state1, reward1, terminated, truncated, info = env.step(action1, -1)
done = terminated or truncated
rtype = info.get('reward_type', 'unknown')
print(f"✓ step() works: reward={reward1}, done={done}, rtype={rtype}")

# Random move
action2 = RL.play_random(state1, env.action_space)
print(f"✓ play_random() works: action = {action2}")

state2, reward2, terminated, truncated, info = env.step(action2, 1)
done = terminated or truncated
rtype = info.get('reward_type', 'unknown')
print(f"✓ second step() works: reward={reward2}, done={done}, rtype={rtype}")

# Test replay buffer
RL.replay_buffer.push(state, action1, reward1, state1, done)
print(f"✓ replay_buffer.push() works: buffer size = {len(RL.replay_buffer)}")

if len(RL.replay_buffer) >= 1:
    print("✓ replay_buffer has enough samples")

print("\n✓✓✓ Config 0 is fully working with gym 0.26+! ✓✓✓")
