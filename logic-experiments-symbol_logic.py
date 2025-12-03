#!/usr/bin/env python3
"""
Set-Based Symbolic Logic Network for Tic-Tac-Toe

ARCHITECTURE PRINCIPLES:
1. Pure symbolic logic - all proposition elements are discrete symbols (embeddings)
2. Set-based working memory - permutation-invariant processing
3. Recurrent forward chaining - rules fire iteratively, building up conclusions
4. Higher-order logic emerges from composition, not arithmetic

PROPOSITION FORMAT:
Each proposition is a tuple of L symbols: (predicate, arg1, arg2, ..., argN)
Examples:
  ('at', 'X', 'TL', 'null')           # X occupies top-left
  ('line', 'row0', 'TL', 'TM')        # Row 0 contains TL and TM
  ('pair', 'X', 'row0', 'null')       # Two X's detected in row0
  ('win_move', 'TR', 'null', 'null')  # Winning move available at TR

SYMBOLIC VOCABULARY:
- Predicates: at, line, corner, center, edge, opposite, pair, threat, fork, win_move
- Players: X, O, empty
- Positions: TL, TM, TR, ML, MM, MR, BL, BM, BR
- Lines: row0, row1, row2, col0, col1, col2, diag0, diag1
- Special: null (padding)

ITERATION EXAMPLE (Win Detection):
Iteration 0: 
  WM = {('at','X','TL'), ('at','X','TM'), ('at','empty','TR'), 
        ('line','row0','TL','TM'), ('line','row0','TM','TR'), ...}

Iteration 1: Rules detect pieces in lines
  WM += {('in_line','X','row0','TL'), ('in_line','X','row0','TM')}

Iteration 2: Rules detect pairs
  WM += {('pair','X','row0')}

Iteration 3: Rules find empty in paired lines
  WM += {('win_move','TR')}

Output: Q-values boosted for positions with ('win_move', pos)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# SYMBOLIC VOCABULARY
# ============================================================================

SYMBOL_VOCAB = {
    # Predicates
    'at': 0, 'line': 1, 'corner': 2, 'center': 3, 'edge': 4, 
    'opposite': 5, 'in_line': 6, 'pair': 7, 'threat': 8, 
    'fork': 9, 'win_move': 10, 'block_move': 11,
    
    # Players
    'X': 12, 'O': 13, 'empty': 14,
    
    # Positions (9 squares)
    'TL': 15, 'TM': 16, 'TR': 17,
    'ML': 18, 'MM': 19, 'MR': 20,
    'BL': 21, 'BM': 22, 'BR': 23,
    
    # Lines (8 lines)
    'row0': 24, 'row1': 25, 'row2': 26,
    'col0': 27, 'col1': 28, 'col2': 29,
    'diag0': 30, 'diag1': 31,
    
    # Special
    'null': 32,
}

VOCAB_SIZE = len(SYMBOL_VOCAB)
POSITION_SYMBOLS = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']
LINE_SYMBOLS = ['row0', 'row1', 'row2', 'col0', 'col1', 'col2', 'diag0', 'diag1']

# Reverse lookup
SYMBOL_NAMES = {v: k for k, v in SYMBOL_VOCAB.items()}

# ============================================================================
# STRUCTURAL KNOWLEDGE (constant across all games)
# ============================================================================

def get_structural_propositions():
    """
    Return fixed structural knowledge about TicTacToe board.
    These are added to every game state's working memory.
    """
    props = []
    
    # Line definitions (which positions belong to each line)
    line_defs = [
        ('row0', ['TL', 'TM', 'TR']),
        ('row1', ['ML', 'MM', 'MR']),
        ('row2', ['BL', 'BM', 'BR']),
        ('col0', ['TL', 'ML', 'BL']),
        ('col1', ['TM', 'MM', 'BM']),
        ('col2', ['TR', 'MR', 'BR']),
        ('diag0', ['TL', 'MM', 'BR']),
        ('diag1', ['TR', 'MM', 'BL']),
    ]
    
    for line_name, positions in line_defs:
        # Each line has 3 positions, store as separate binary relations
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                props.append(('line', line_name, positions[i], positions[j]))
    
    # Position type predicates
    corners = ['TL', 'TR', 'BL', 'BR']
    for c in corners:
        props.append(('corner', c, 'null', 'null'))
    
    props.append(('center', 'MM', 'null', 'null'))
    
    edges = ['TM', 'ML', 'MR', 'BM']
    for e in edges:
        props.append(('edge', e, 'null', 'null'))
    
    # Opposite corner relations
    props.append(('opposite', 'TL', 'BR', 'null'))
    props.append(('opposite', 'BR', 'TL', 'null'))
    props.append(('opposite', 'TR', 'BL', 'null'))
    props.append(('opposite', 'BL', 'TR', 'null'))
    
    return props


def board_to_symbolic_wm(state_vector):
    """
    Convert state vector to symbolic working memory.
    
    Args:
        state_vector: Numpy array from environment (could be plain board or logic encoding)
    
    Returns:
        List of symbolic propositions
    """
    wm = []
    
    # If state is logic encoding (length 18: 9 positions × 2 features each)
    # Extract just the player values
    if len(state_vector) == 18:
        board = []
        for i in range(0, 18, 2):
            player_val = state_vector[i]
            # Convert to 1=X, -1=O, 0=empty
            if player_val > 0.5:
                board.append(1)
            elif player_val < -0.5:
                board.append(-1)
            else:
                board.append(0)
    else:
        # Assume it's already a 9-element board
        board = state_vector.tolist() if hasattr(state_vector, 'tolist') else list(state_vector)
    
    # Convert board to propositions
    for i, value in enumerate(board[:9]):  # Ensure only 9 squares
        pos_symbol = POSITION_SYMBOLS[i]
        
        if value == 1:
            player = 'X'
        elif value == -1:
            player = 'O'
        else:
            player = 'empty'
        
        wm.append(('at', player, pos_symbol, 'null'))
    
    # Add structural knowledge
    wm.extend(get_structural_propositions())
    
    return wm


def propositions_to_tensor(props, L=4):
    """
    Convert list of symbolic propositions to tensor of symbol indices.
    
    Args:
        props: List of tuples like ('at', 'X', 'TL', 'null')
        L: Length of each proposition (padding with 'null')
    
    Returns:
        Tensor of shape [num_props, L] with symbol indices
    """
    tensor_props = []
    
    for prop in props:
        # Pad or truncate to length L
        padded = list(prop) + ['null'] * (L - len(prop))
        padded = padded[:L]
        
        # Convert symbols to indices
        indices = [SYMBOL_VOCAB[sym] for sym in padded]
        tensor_props.append(indices)
    
    return torch.tensor(tensor_props, dtype=torch.long)


# ============================================================================
# SET-BASED SYMBOLIC LOGIC NETWORK
# ============================================================================

class SymbolicLogicNetwork(nn.Module):
    """
    Recurrent logic network with pure symbolic reasoning.
    
    Architecture:
    - Embedding layer: Maps each symbol to learned vector
    - M logic rules: Each can match patterns and generate conclusions
    - Recurrent iterations: Rules fire T times, building up WM
    - Output layer: Maps final WM to Q-values for actions
    """
    
    def __init__(self, M=6, J=2, I=3, L=4, embed_dim=4, max_iters=2):
        super(SymbolicLogicNetwork, self).__init__()
        
        self.M = M  # Number of rules (reduced from 12)
        self.J = J  # Premises per rule (reduced from 3)
        self.I = I  # Variable slots per rule (reduced from 4)
        self.L = L  # Proposition length (predicate + args)
        self.embed_dim = embed_dim  # Minimal for 33 symbols (was 8)
        self.max_iters = max_iters  # Reduced from 3
        
        print(f"Symbolic Logic Network: {M} rules, {J} premises, {I} variables, {max_iters} iterations")
        
        # Symbol embedding layer (LEARNED during training)
        self.symbol_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        
        # Initialize with small random values
        # These will be updated via backprop to capture symbol semantics
        nn.init.normal_(self.symbol_embedding.weight, mean=0, std=0.1)
        
        # Logic rules
        self.rules = nn.ModuleList()
        for m in range(self.M):
            rule = nn.Module()
            
            # Each premise: L positions, each is a symbol index (NOT learned, just template)
            # These are fixed integer indices, not parameters
            rule.register_buffer('constants', 
                torch.randint(0, VOCAB_SIZE, (self.J, self.L)))
            
            # Gammas control constant vs variable (THESE are learned)
            rule.gammas = nn.Parameter(torch.rand(self.J, self.L))
            
            # Variable capture layers (learned)
            rule.capture = nn.ModuleList([
                nn.Linear(embed_dim * self.L, self.I * embed_dim)
                for _ in range(self.J)
            ])
            
            # Conclusion generation (learned)
            rule.conclude = nn.Linear(self.I * embed_dim * self.J, self.L * VOCAB_SIZE)
            
            self.rules.append(rule)
        
        # Output layer: Map WM to Q-values
        self.output_layer = nn.Linear(embed_dim * self.L, 9)
        
    def forward(self, state):
        """
        Forward pass with recurrent logic iterations.
        
        Args:
            state: Numeric board state [batch, 9] with values 1=X, -1=O, 0=empty
        
        Returns:
            Q-values for 9 positions [batch, 9]
        """
        batch_size = state.shape[0]
        
        # Convert numeric state to symbolic WM
        wm_list = []
        for b in range(batch_size):
            board = state[b].cpu().numpy() if state.is_cuda else state[b].numpy()
            board = board.astype(int).tolist()
            
            # Convert to symbolic propositions
            props = board_to_symbolic_wm(board)
            wm_list.append(props)
        
        # Convert to tensor [batch, max_num_props, L]
        max_props = max(len(wm) for wm in wm_list)
        
        wm_tensor = torch.zeros(batch_size, max_props, self.L, dtype=torch.long).to(device)
        for b, props in enumerate(wm_list):
            prop_tensor = propositions_to_tensor(props, self.L)
            wm_tensor[b, :len(props)] = prop_tensor.to(device)
        
        # Recurrent iterations (with early stopping)
        for t in range(self.max_iters):
            old_size = wm_tensor.shape[1]
            wm_tensor = self.apply_rules(wm_tensor)
            
            # Limit WM growth to prevent explosion
            if wm_tensor.shape[1] > 100:
                wm_tensor = wm_tensor[:, :100, :]
            
            # Early stop if no new props added
            if wm_tensor.shape[1] == old_size:
                break
        
        # Extract Q-values from final WM
        q_values = self.wm_to_qvalues(wm_tensor)
        
        return q_values
    
    def apply_rules(self, wm):
        """
        Apply all rules to current working memory.
        
        Args:
            wm: [batch, num_props, L] symbol indices
        
        Returns:
            Updated WM with new conclusions added
        """
        batch_size = wm.shape[0]
        num_props = wm.shape[1]
        
        # Embed all propositions
        wm_embedded = self.symbol_embedding(wm)  # [batch, num_props, L, embed_dim]
        
        new_conclusions = []
        
        for rule in self.rules:
            # Match premises against WM
            conclusions = self.match_and_fire(rule, wm_embedded)
            if conclusions is not None:
                new_conclusions.append(conclusions)
        
        # Add new conclusions to WM
        if new_conclusions:
            new_wm = torch.cat([wm] + new_conclusions, dim=1)
        else:
            new_wm = wm
        
        return new_wm
    
    def match_and_fire(self, rule, wm_embedded):
        """
        Match rule premises against WM and generate conclusions.
        Uses soft attention for permutation-invariant set matching.
        
        Args:
            rule: Rule module
            wm_embedded: [batch, num_props, L, embed_dim]
        
        Returns:
            New propositions [batch, num_new_props, L] or None
        """
        batch_size = wm_embedded.shape[0]
        num_props = wm_embedded.shape[1]
        
        # Embed rule's constant templates
        rule_constants_embedded = self.symbol_embedding(rule.constants)  # [J, L, embed_dim]
        
        # Sigmoid activation for gamma (0=constant, 1=variable)
        gamma = torch.sigmoid(rule.gammas)  # [J, L]
        
        # For each premise, compute match scores against all WM propositions
        premise_matches = []  # List of [batch, num_props] tensors
        captured_variables = []  # List of [batch, num_props, I*embed_dim] tensors
        
        for j in range(self.J):
            # Premise j's constants and gammas
            const_j = rule_constants_embedded[j]  # [L, embed_dim]
            gamma_j = gamma[j]  # [L]
            
            # Match scores: How well does each WM prop match this premise?
            # [batch, num_props, L, embed_dim]
            wm_props = wm_embedded
            
            # Compute element-wise distance
            # Expand const_j to [1, 1, L, embed_dim] for broadcasting
            const_expanded = const_j.unsqueeze(0).unsqueeze(0)  # [1, 1, L, embed_dim]
            gamma_expanded = gamma_j.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, L, 1]
            
            # Distance in embedding space
            dist = torch.norm(wm_props - const_expanded, dim=-1)  # [batch, num_props, L]
            
            # Weighted by gamma: constant positions contribute to mismatch
            weighted_dist = (1 - gamma_expanded.squeeze(-1)) * dist  # [batch, num_props, L]
            
            # Total match score per proposition (lower is better)
            match_scores = weighted_dist.sum(dim=-1)  # [batch, num_props]
            
            # Convert to attention weights (higher for better matches)
            attention = F.softmax(-match_scores / 0.1, dim=1)  # [batch, num_props]
            
            premise_matches.append(attention)
            
            # Capture variables from matched propositions
            # Weighted sum of WM propositions based on attention
            wm_flat = wm_props.reshape(batch_size, num_props, -1)  # [batch, num_props, L*embed_dim]
            captured = rule.capture[j](wm_flat)  # [batch, num_props, I*embed_dim]
            
            # Weight by attention
            weighted_captured = attention.unsqueeze(-1) * captured  # [batch, num_props, I*embed_dim]
            captured_sum = weighted_captured.sum(dim=1)  # [batch, I*embed_dim]
            
            captured_variables.append(captured_sum)
        
        # Combine captured variables from all premises
        all_captured = torch.cat(captured_variables, dim=1)  # [batch, J*I*embed_dim]
        
        # Generate conclusion (new proposition symbols)
        conclusion_logits = rule.conclude(all_captured)  # [batch, L*VOCAB_SIZE]
        conclusion_logits = conclusion_logits.reshape(batch_size, self.L, VOCAB_SIZE)
        
        # Sample or take argmax for discrete symbols
        # Use straight-through estimator (faster than Gumbel-Softmax)
        conclusion_probs = F.softmax(conclusion_logits, dim=-1)  # [batch, L, VOCAB_SIZE]
        conclusion_indices = conclusion_probs.argmax(dim=-1)  # [batch, L]
        
        # Compute firing strength (rule should only fire if all premises match well)
        # Minimum attention across premises
        firing_strength = torch.stack(premise_matches, dim=1).min(dim=1)[0].max(dim=1)[0]  # [batch]
        
        # Threshold: only add conclusions if rule fires strongly
        threshold = 0.1
        mask = firing_strength > threshold  # [batch]
        
        if mask.sum() == 0:
            return None
        
        # Return new propositions for batch items where rule fired
        return conclusion_indices.unsqueeze(1)  # [batch, 1, L]
    
    def wm_to_qvalues(self, wm):
        """
        Convert working memory to Q-values for actions.
        Looks for special propositions like ('win_move', pos, ...) that indicate good moves.
        
        Args:
            wm: [batch, num_props, L] symbol indices
        
        Returns:
            Q-values [batch, 9]
        """
        batch_size = wm.shape[0]
        
        # Initialize Q-values
        q_values = torch.zeros(batch_size, 9).to(wm.device)
        
        # Embed WM propositions
        wm_embedded = self.symbol_embedding(wm)  # [batch, num_props, L, embed_dim]
        
        # Look for action-indicating propositions
        # E.g., ('win_move', 'TL', ...) should boost Q-value for TL
        
        win_move_idx = SYMBOL_VOCAB['win_move']
        block_move_idx = SYMBOL_VOCAB['block_move']
        
        for b in range(batch_size):
            for p in range(wm.shape[1]):
                pred = wm[b, p, 0].item()
                
                # Check if this is a win_move or block_move proposition
                if pred == win_move_idx:
                    # Extract position from 2nd argument
                    pos_symbol_idx = wm[b, p, 1].item()
                    pos_name = SYMBOL_NAMES.get(pos_symbol_idx, 'null')
                    
                    if pos_name in POSITION_SYMBOLS:
                        pos_idx = POSITION_SYMBOLS.index(pos_name)
                        q_values[b, pos_idx] += 10.0  # Strong positive signal
                
                elif pred == block_move_idx:
                    pos_symbol_idx = wm[b, p, 1].item()
                    pos_name = SYMBOL_NAMES.get(pos_symbol_idx, 'null')
                    
                    if pos_name in POSITION_SYMBOLS:
                        pos_idx = POSITION_SYMBOLS.index(pos_name)
                        q_values[b, pos_idx] += 5.0  # Medium positive signal
        
        # Also use learned projection from WM embeddings (for smoother learning)
        wm_flat = wm_embedded.reshape(batch_size, wm.shape[1], -1)  # [batch, num_props, L*embed_dim]
        wm_mean = wm_flat.mean(dim=1)  # [batch, L*embed_dim]
        
        learned_q = self.output_layer(wm_mean)  # [batch, 9]
        
        # Combine rule-based and learned Q-values
        q_values = q_values + 0.1 * learned_q
        
        return q_values


# ============================================================================
# DQN WRAPPER AND TRAINING
# ============================================================================

class ReplayBuffer:
    """Standard experience replay buffer"""
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
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQN:
    """DQN agent with symbolic logic network"""
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-network and target network
        self.q_network = SymbolicLogicNetwork().to(device)
        self.target_network = SymbolicLogicNetwork().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(10000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Endstate for terminal transitions
        self.endState = np.zeros(state_dim)
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size):
        """Train on a batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss and update
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def sync(self):
        """Sync target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def play_random(self, state, action_space):
        """Random action for opponent"""
        return action_space.sample()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, './gym-tictactoe')
    from gym_tictactoe.TTT_logic_dim2_uniform import TicTacToeEnv
    
    print("Symbolic Logic Network for Tic-Tac-Toe")
    print("=" * 60)
    
    # Test symbol vocabulary
    print(f"\nVocabulary size: {VOCAB_SIZE}")
    print(f"Sample symbols: {list(SYMBOL_VOCAB.keys())[:10]}")
    
    # Test structural propositions
    struct_props = get_structural_propositions()
    print(f"\nStructural propositions: {len(struct_props)}")
    print(f"Sample: {struct_props[:3]}")
    
    # Test board conversion
    test_board = [1, 1, 0, 0, -1, 0, 0, -1, 0]  # X X _ / _ O _ / _ O _
    wm = board_to_symbolic_wm(test_board)
    print(f"\nWorking memory size: {len(wm)}")
    print(f"Board propositions: {wm[:9]}")
    
    # Test tensor conversion
    wm_tensor = propositions_to_tensor(wm)
    print(f"\nWM tensor shape: {wm_tensor.shape}")
    print(f"First proposition indices: {wm_tensor[0]}")
    print(f"Decoded: {[SYMBOL_NAMES[idx.item()] for idx in wm_tensor[0]]}")
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Create environment
    env = TicTacToeEnv(symbols=[-1, 1], board_size=3, win_size=3)
    
    # Create agent
    RL = DQN(
        state_dim=env.state_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=0.001
    )
    
    # Training parameters
    num_episodes = 10000  # Longer run to see learning
    batch_size = 32  # Smaller batches
    sync_every = 50
    print_every = 100
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        max_steps = 10  # Prevent long episodes
        
        while not done and episode_length < max_steps:
            # Agent chooses action
            action = RL.choose_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action, symbol=1)
            done = terminated or truncated
            
            # Store transition
            RL.replay_buffer.push(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Train every 4 steps
            if episode_length % 4 == 0 and len(RL.replay_buffer) >= batch_size:
                loss = RL.update(batch_size)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Sync target network
        if episode % sync_every == 0:
            RL.sync()
        
        # Decay epsilon
        RL.decay_epsilon()
        
        # Print progress
        if episode % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:]) if len(episode_rewards) >= print_every else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-print_every:]) if len(episode_lengths) >= print_every else np.mean(episode_lengths)
            
            print(f"Episode {episode:5d} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Length: {avg_length:4.1f} | "
                  f"Epsilon: {RL.epsilon:.4f}")
    
    print("\nTraining complete!")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
