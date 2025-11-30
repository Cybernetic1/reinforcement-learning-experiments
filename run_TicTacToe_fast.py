#!/usr/bin/env python3
"""
Optimized TicTacToe training for faster convergence
Based on config 28 (DQN with logic-with-vars)
"""

import sys
sys.path.insert(0, './gym-tictactoe')
import gym_tictactoe
from gym_tictactoe.TTT_logic_dim2_uniform import TicTacToeEnv
from DQN_logic_with_vars import DQN, ReplayBuffer
import time

# Create environment
env = TicTacToeEnv(symbols=[-1, 1], board_size=3, win_size=3)
env.seed(333)

# Create agent with optimized hyperparameters
RL = DQN(
    action_dim=env.action_space.n,
    state_dim=env.state_space.shape[0],
    learning_rate=0.003,  # Increased from 0.001 for faster learning
    gamma=0.9,
)

# Optimized hyperparameters for faster convergence
batch_size = 128  # Smaller batch for faster updates
reward_scale = 10.0
update_frequency = 1  # Update every game instead of every N games
target_sync_frequency = 50  # Sync target network more frequently

print("=" * 70)
print("FAST TRAINING MODE - Optimizations:")
print("- M=8 rules (doubled capacity for learning strategies)")
print("- Learning rate: 0.003 (3x normal)")
print("- Batch size: 128 (smaller for faster updates)")
print("- Update every game")
print("- Target network sync every 50 episodes")
print("- No rendering/visualization")
print("=" * 70)

i_episode = 0
running_reward = 0.0

try:
    while True:
        i_episode += 1
        state, _ = env.reset()
        done = False
        user = -1
        reward1 = 0
        reward2 = 0

        while not done:
            if user == -1:  # AI player
                action1 = RL.choose_action(state)
                state1, reward1, terminated, truncated, info = env.step(action1, -1)
                done = terminated or truncated
                if done:
                    RL.replay_buffer.push(state, action1, reward1, RL.endState, done)
            elif user == 1:  # random player
                action2 = RL.play_random(state1, env.action_space)
                state2, reward2, terminated, truncated, info = env.step(action2, 1)
                done = terminated or truncated
                
                r_x = reward1  # reward w.r.t. player X = AI
                if reward2 > 19.0:
                    r_x -= 20.0
                elif reward2 > 9.0:  # draw
                    r_x += 10.0
                
                if done:
                    RL.replay_buffer.push(state, action1, r_x, RL.endState, done)
                else:
                    RL.replay_buffer.push(state, action1, r_x, state2, done)
                state = state2

            # Change player
            if not done:
                rtype = info.get('reward_type', 'unknown')
                if rtype != 'thinking':
                    user = -1 if user == 1 else 1

        # Update running reward
        per_game_reward = RL.replay_buffer.last_reward()
        running_reward = running_reward * 0.97 + per_game_reward * 0.03

        # More frequent updates
        if len(RL.replay_buffer) > batch_size:
            loss = RL.update(batch_size, reward_scale)
        else:
            loss = None

        # Decay epsilon
        RL.decay_epsilon()

        # Sync target network more frequently
        if i_episode % target_sync_frequency == 0:
            RL.sync()
            
            rr = round(running_reward, 5)
            print(f"Episode {i_episode:5d} | Reward: {rr:7.3f} | Epsilon: {RL.epsilon:.4f}", end="")
            if loss is not None:
                print(f" | Loss: {loss:.4f}", end="")
            print()

        # Save checkpoint every 5000 episodes
        if i_episode % 5000 == 0:
            timestamp = time.strftime("%d-%m-%Y(%H:%M)")
            config, n_params = RL.net_info()
            fname = f"model.DQN.logic_with_vars.AlgebraicLogic.{config}.{timestamp}"	# no need to add .dict
            RL.save_net(fname)
            print(f"✓ Model saved: {fname}")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    timestamp = time.strftime("%d-%m-%Y(%H:%M)")
    config, n_params = RL.net_info()
    fname = f"model.DQN.logic_with_vars.AlgebraicLogic.{config}.{timestamp}.dict"
    RL.save_net(fname)
    print(f"✓ Final model saved: {fname}")
