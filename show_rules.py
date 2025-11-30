#!/usr/bin/env python3
"""
Display learned rules from trained model
"""

import sys
sys.path.insert(0, './gym-tictactoe')

import torch
import torch.nn.functional as F
from DQN_logic_with_vars import DQN
from gym_tictactoe.TTT_logic_dim2_uniform import TicTacToeEnv

# Create environment and agent
env = TicTacToeEnv(symbols=[-1, 1], board_size=3, win_size=3)
RL = DQN(
    action_dim=env.action_space.n,
    state_dim=env.state_space.shape[0],
    learning_rate=0.001,
    gamma=0.9,
)

# Load the trained model
import glob
files = glob.glob("PyTorch_models/*M=8*.dict")
if not files:
    print("No M=8 models found, trying M=4...")
    files = glob.glob("PyTorch_models/*M=4*.dict")
    
if not files:
    print("No trained models found!")
    sys.exit(1)

files.sort()
print(f"Found {len(files)} models. Using latest: {files[-1]}")
RL.load_net(files[-1][15:-5])  # Remove "PyTorch_models/" and ".dict"

print("\n" + "=" * 80)
print("LEARNED RULES ANALYSIS")
print("=" * 80)

net = RL.anet
sigmoid = torch.nn.Sigmoid()

# Board position names for clarity
pos_names = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']

for rule_idx, rule in enumerate(net.rules):
    print(f"\n{'=' * 80}")
    print(f"RULE {rule_idx + 1} (of {net.M})")
    print(f"{'=' * 80}")
    
    # Analyze each premise
    print(f"\nPREMISES (IF part - what to look for on board):")
    for j in range(net.J):
        print(f"\n  Premise {j+1}:")
        
        # Get gamma values (constant vs variable mode)
        gammas = sigmoid(rule.γs[j]).detach()
        constants = rule.constants[j].detach()
        
        for l in range(net.L):
            gamma_val = gammas[l].item()
            const_val = constants[l].item()
            
            if l == 0:  # Player dimension
                if gamma_val < 0.3:  # Constant mode
                    player_str = "X" if const_val > 0 else "O" if const_val < -0.5 else "empty"
                    print(f"    Element {l} (player): MATCH {player_str} (γ={gamma_val:.3f}, constant={const_val:.2f})")
                else:
                    print(f"    Element {l} (player): CAPTURE any (γ={gamma_val:.3f})")
            else:  # Position dimension
                if gamma_val < 0.3:  # Constant mode
                    # Position encoding: -1 to +1 maps to positions 0-8
                    pos_idx = int((const_val + 1) * 4.5)
                    pos_idx = max(0, min(8, pos_idx))
                    print(f"    Element {l} (position): MATCH {pos_names[pos_idx]} (γ={gamma_val:.3f}, constant={const_val:.2f})")
                else:
                    print(f"    Element {l} (position): CAPTURE any (γ={gamma_val:.3f})")
    
    # Analyze the head (THEN part - what action to take)
    print(f"\nCONCLUSION (THEN part - Q-values for each position):")
    print(f"  Rule head weights shape: {rule.head.weight.shape}")
    
    # To understand what the rule concludes, we need to see what it outputs
    # Let's look at the head weights (maps I variables → W outputs)
    head_weights = rule.head.weight.detach()  # [W=9, I=3]
    head_bias = rule.head.bias.detach()  # [W=9]
    
    # Unicode subscripts for numbers 0-9
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    
    print(f"\n  Q-value contributions by position:")
    for pos_idx in range(net.W):
        bias = head_bias[pos_idx].item()
        weights = head_weights[pos_idx]
        
        # Build polynomial-style expression
        terms = []
        for i in range(net.I):
            coef = weights[i].item()
            var_name = f"x{i}".translate(subscripts)
            if coef >= 0:
                terms.append(f"{coef:.2f} {var_name}")
            else:
                terms.append(f"{coef:.2f} {var_name}")
        
        expr = " + ".join(terms)
        if bias >= 0:
            expr += f" + {bias:.2f}"
        else:
            expr += f" {bias:.2f}"
        
        print(f"    {pos_names[pos_idx]}: {expr}")
    
    # Identify most favored positions (highest bias)
    top_positions = torch.topk(head_bias, k=3)
    print(f"\n  Top 3 favored positions (by bias):")
    for i, (val, idx) in enumerate(zip(top_positions.values, top_positions.indices)):
        print(f"    {i+1}. {pos_names[idx.item()]}: bias = {val.item():.2f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total rules: {net.M}")
print(f"Premises per rule: {net.J}")
print(f"Variable slots per rule: {net.I}")
print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
print("\nNote: Rules work together - the final Q-value for each position is the")
print("sum of contributions from all rules, weighted by their match confidence.")
