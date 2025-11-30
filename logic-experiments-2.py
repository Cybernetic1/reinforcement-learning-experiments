#!/usr/bin/env python3
"""
Copilot question: We know that TicTacToe has some symmetries and by config 2 we know that this symmetry greatly accelerated learning.  So the rules with variables potentially can reduce the number of rules, but they fail to emerge in the initial stages.  It seems that the process of learning concrete rules is a prerequisite for learning general rules...?  But the idea of deep learning seems to be that high-level and low-level rules can be learned at the same time -- we just need to stack them up via layer composition.  Perhaps this is lacking in our current setup?

Logic Experiments 2: Two-layer hierarchical rule architecture
- Layer 1: Concrete pattern detectors (γ=0, forced constant mode)
- Layer 2: Variable-based reasoning that composes Layer 1 patterns
"""

import sys
sys.path.insert(0, './gym-tictactoe')
import gym_tictactoe
from gym_tictactoe.TTT_logic_dim2_uniform import TicTacToeEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HierarchicalAlgelogicNetwork(nn.Module):
    """
    Two-layer rule architecture:
    1. Layer 1 (concrete): M1 rules with forced constant matching (γ=0)
    2. Layer 2 (abstract): M2 rules with variables that use Layer 1 outputs
    """
    
    def __init__(self, input_dim, action_dim, M1=8, M2=4, learning_rate=3e-3):
        super(HierarchicalAlgelogicNetwork, self).__init__()
        
        # Architecture parameters
        self.M1 = M1  # Number of Layer 1 rules (concrete patterns)
        self.M2 = M2  # Number of Layer 2 rules (abstract composition)
        self.J = 2    # Premises per rule
        self.I = 3    # Variables per rule
        self.L = 2    # Proposition length (player, position)
        self.W = 9    # Output positions
        
        print(f"Architecture: {M1} concrete rules → {M2} abstract rules")
        
        # LAYER 1: Concrete pattern detectors
        self.concrete_rules = nn.ModuleList()
        for m in range(self.M1):
            rule = nn.Module()
            
            # Constants for pattern matching (γ=0, fixed)
            cs = torch.FloatTensor(self.J, self.L).uniform_(-1, 1)
            rule.constants = nn.Parameter(cs)
            
            # No variables in Layer 1 - just pattern detection
            # Output: single confidence score for this pattern
            rule.detector = nn.Linear(1, 1)  # Confidence from match score
            
            self.concrete_rules.append(rule)
        
        # LAYER 2: Abstract rules with variables
        self.abstract_rules = nn.ModuleList()
        for m in range(self.M2):
            rule = nn.Module()
            
            # Process Layer 1 outputs (M1 pattern confidences) + original state
            # Input: [batch, M1 + W*L] → captures both patterns and raw board
            input_size = self.M1 + self.W * self.L
            
            # Variable capture from combined representation
            rule.body = nn.ModuleList()
            for j in range(self.J):
                rule.body.append(nn.Linear(input_size, self.I))
            
            # γ parameters: decide which Layer 1 patterns to use as variables
            γs = torch.FloatTensor(self.J).uniform_(0, 1)
            rule.γs = nn.Parameter(γs)
            
            # Head: map captured variables to Q-values
            rule.head = nn.Linear(self.I * self.J, self.W)
            
            self.abstract_rules.append(rule)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, state):
        """
        state: [batch, W*L=18] board representation
        """
        batch_size = state.shape[0]
        state_2d = state.view(batch_size, self.W, self.L)  # [batch, 9, 2]
        
        # === LAYER 1: Detect concrete patterns ===
        pattern_scores = torch.zeros(batch_size, self.M1).to(device)
        
        for m, rule in enumerate(self.concrete_rules):
            # For each premise, compute match score
            match_score = torch.zeros(batch_size).to(device)
            
            for j in range(self.J):
                # Find best matching position for this premise
                # Match against constant (γ=0 forced)
                constant = rule.constants[j]  # [L=2]
                
                # Compute distance to each board position
                for l in range(self.L):
                    diff = (constant[l] - state_2d[:, :, l]) ** 2
                    match_score += diff.min(dim=1)[0]  # Best match across positions
            
            # Convert match score to confidence (lower is better)
            confidence = torch.exp(-match_score / 10.0)  # Temperature scaling
            pattern_scores[:, m] = rule.detector(confidence.unsqueeze(1)).squeeze()
        
        # === LAYER 2: Compose patterns with variables ===
        outputs = torch.zeros(batch_size, self.W).to(device)
        
        # Combine Layer 1 outputs with original state
        combined = torch.cat([pattern_scores, state], dim=1)  # [batch, M1+18]
        
        for rule in self.abstract_rules:
            # Capture variables from combined representation
            captured_vars = []
            
            for j in range(self.J):
                vars_j = rule.body[j](combined)  # [batch, I]
                γ_j = self.sigmoid(rule.γs[j])
                captured_vars.append(γ_j * vars_j)
            
            # Concatenate all captured variables
            all_vars = torch.cat(captured_vars, dim=1)  # [batch, J*I]
            
            # Generate Q-values
            rule_output = rule.head(all_vars)  # [batch, W]
            outputs += rule_output
        
        return outputs

# Rest of DQN implementation (same as before)
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
    def last_reward(self):
        return self.buffer[self.position - 1][2]

class DQN:
    def __init__(self, action_dim, state_dim, learning_rate=0.001, gamma=0.9):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.anet = HierarchicalAlgelogicNetwork(state_dim, action_dim, M1=8, M2=4, learning_rate=learning_rate).to(device)
        self.tnet = HierarchicalAlgelogicNetwork(state_dim, action_dim, M1=8, M2=4, learning_rate=learning_rate).to(device)
        self.tnet.load_state_dict(self.anet.state_dict())
        
        self.optimizer = torch.optim.Adam(self.anet.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(10000)
        self.endState = [1]*18  # Generic end state marker
    
    def choose_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.anet(state_tensor).squeeze(0)
        
        # Mask invalid actions
        state_2d = state_tensor.view(9, 2)
        valid_mask = (state_2d[:, 0] == 0)
        
        if not valid_mask.any():
            return random.randint(0, 8)
        
        q_values[~valid_mask] = float('-inf')
        
        if not deterministic and random.random() < self.epsilon:
            valid_actions = torch.where(valid_mask)[0]
            return valid_actions[random.randint(0, len(valid_actions)-1)].item()
        
        return torch.argmax(q_values).item()
    
    def play_random(self, state, action_space):
        empties = []
        for i in range(9):
            if state[i*2] == 0:
                empties.append(i)
        return random.choice(empties) if empties else 0
    
    def update(self, batch_size, reward_scale):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        q_values = self.anet(states)
        next_q_values = self.tnet(next_states)
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        
        loss = F.mse_loss(q_value, expected_q_value.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.anet.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def sync(self):
        self.tnet.load_state_dict(self.anet.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_net(self, fname):
        torch.save(self.anet.state_dict(), f"PyTorch_models/{fname}.dict")
        print(f"Model saved to PyTorch_models/{fname}.dict")
    
    def load_net(self, fname):
        self.anet.load_state_dict(torch.load(f"PyTorch_models/{fname}.dict"))
        self.tnet.load_state_dict(self.anet.state_dict())
        print(f"Model loaded from PyTorch_models/{fname}.dict")
    
    def net_info(self):
        total_params = sum(p.numel() for p in self.anet.parameters())
        config = f"Hierarchical.M1={self.anet.M1},M2={self.anet.M2}"
        return (config, total_params)

# Training setup
env = TicTacToeEnv(symbols=[-1, 1], board_size=3, win_size=3)
env.seed(333)

RL = DQN(
    action_dim=env.action_space.n,
    state_dim=env.state_space.shape[0],
    learning_rate=0.001,  # Lower LR for stability
    gamma=0.9,
)

batch_size = 128
reward_scale = 10.0
target_sync_frequency = 50

print("=" * 70)
print("LOGIC EXPERIMENTS 2: Hierarchical Architecture")
print("- Layer 1: 8 concrete pattern detectors (forced constants)")
print("- Layer 2: 4 abstract rules with variables (compose L1)")
print("- Learning rate: 0.001")
print("- Target network sync every 50 episodes")
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
            if user == -1:
                action1 = RL.choose_action(state)
                state1, reward1, terminated, truncated, info = env.step(action1, -1)
                done = terminated or truncated
                if done:
                    RL.replay_buffer.push(state, action1, reward1, RL.endState, done)
            elif user == 1:
                action2 = RL.play_random(state1, env.action_space)
                state2, reward2, terminated, truncated, info = env.step(action2, 1)
                done = terminated or truncated
                
                r_x = reward1
                if reward2 > 19.0:
                    r_x -= 20.0
                elif reward2 > 9.0:
                    r_x += 10.0
                
                if done:
                    RL.replay_buffer.push(state, action1, r_x, RL.endState, done)
                else:
                    RL.replay_buffer.push(state, action1, r_x, state2, done)
                state = state2

            if not done:
                rtype = info.get('reward_type', 'unknown')
                if rtype != 'thinking':
                    user = -1 if user == 1 else 1

        per_game_reward = RL.replay_buffer.last_reward()
        running_reward = running_reward * 0.97 + per_game_reward * 0.03

        if len(RL.replay_buffer) > batch_size:
            loss = RL.update(batch_size, reward_scale)
        else:
            loss = None

        RL.decay_epsilon()

        if i_episode % target_sync_frequency == 0:
            RL.sync()
            
            rr = round(running_reward, 5)
            print(f"Episode {i_episode:5d} | Reward: {rr:7.3f} | Epsilon: {RL.epsilon:.4f}", end="")
            if loss is not None:
                print(f" | Loss: {loss:.4f}", end="")
            print()

        if i_episode % 1000 == 0:
            timestamp = time.strftime("%d-%m-%Y(%H:%M)")
            config, n_params = RL.net_info()
            fname = f"model.DQN.hierarchical.{config}.{timestamp}"
            RL.save_net(fname)
            print(f"✓ Model saved: {fname}")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    timestamp = time.strftime("%d-%m-%Y(%H:%M)")
    config, n_params = RL.net_info()
    fname = f"model.DQN.hierarchical.{config}.{timestamp}"
    RL.save_net(fname)
    print(f"✓ Final model saved: {fname}")
