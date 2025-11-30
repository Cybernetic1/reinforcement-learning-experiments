#!/usr/bin/env python3
"""Plot training progress - robust version"""

import matplotlib.pyplot as plt
import numpy as np
import re

episodes = []
rewards = []

# Parse data more robustly
with open('training_data.txt', 'rb') as f:
    for line in f:
        try:
            line_str = line.decode('utf-8', errors='ignore').strip()
            parts = line_str.split()
            if len(parts) >= 2:
                ep = float(parts[0])
                rw = float(parts[1])
                episodes.append(ep)
                rewards.append(rw)
        except:
            continue

episodes = np.array(episodes)
rewards = np.array(rewards)

print(f'Loaded {len(episodes)} data points')

# Create plot
plt.figure(figsize=(14, 7))
plt.plot(episodes, rewards, alpha=0.5, linewidth=0.8, label='Running Reward', color='blue')

# Add moving average
window = 50
if len(rewards) >= window:
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2.5, label=f'Moving Avg ({window} episodes)')

plt.xlabel('Episode', fontsize=13)
plt.ylabel('Running Reward', fontsize=13)
plt.title('TicTacToe Training Progress (Fast Mode - ~11 hours, 30k episodes)', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Win threshold')

# Add statistics
final_reward = rewards[-1]
best_reward = rewards.max()
avg_last_100 = rewards[-100:].mean() if len(rewards) >= 100 else rewards.mean()
avg_last_1000 = rewards[-1000:].mean() if len(rewards) >= 1000 else rewards.mean()

stats_text = f'Final: {final_reward:.2f}\nBest: {best_reward:.2f}\nAvg (last 100): {avg_last_100:.2f}\nAvg (last 1000): {avg_last_1000:.2f}'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150)
print(f'\n✓ Plot saved to training_progress.png')
print(f'\nTraining Statistics:')
print(f'  Episodes: {int(episodes[-1])}')
print(f'  Duration: ~11 hours')
print(f'  Final reward: {final_reward:.3f}')
print(f'  Best reward: {best_reward:.3f}')
print(f'  Average (last 1000): {avg_last_1000:.3f}')
print(f'  Average (last 100): {avg_last_100:.3f}')
print(f'  Average (all): {rewards.mean():.3f}')
print(f'  Improvement from start: {rewards[-100:].mean() - rewards[:100].mean():.3f}')
print(f'  Std dev (last 1000): {rewards[-1000:].std():.3f}')
