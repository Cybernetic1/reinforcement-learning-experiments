"""
logic.hierarchical module - 3-layer Logic Network architecture for TicTacToe.

CORE INNOVATION: 3-layer hierarchical logical reasoning system
1. CONCRETE LAYER: Low-level pattern detection (no variables)
2. ABSTRACTION LAYER: Strategic concept formation (with variables)  
3. ACTION LAYER: Direct concept-to-action mapping (no variables)

ARCHITECTURE OVERVIEW:
Layer 1: Board propositions [9×2] → Concrete features [12×2]
Layer 2: Concrete features [12×2] → Abstract concepts [6×2] 
Layer 3: Abstract concepts [6×2] → Q-values [9×1]

BOARD REPRESENTATION:
- Each square encoded as [player, normalized_position]
- player ∈ {-1, 0, 1} for {opponent, empty, self}
- position ∈ [-1, +1] using formula (square_index - 4)/4
- Examples:
  * Square 0 (top-left): [player, -1.0]
  * Square 4 (center): [player, 0.0] 
  * Square 8 (bottom-right): [player, +1.0]

PROPOSITION SEMANTICS:
- Concrete features: [feature_strength, spatial_focus]
  Example: [0.8, -1.0] = "Strong empty corner pattern at top-left"
- Abstract concepts: [strategic_value, action_preference]
  Example: [0.7, -1.0] = "High fork opportunity, prefer corners"

RESEARCH HYPOTHESIS: Hierarchical logical reasoning with clear separation
between perception, cognition, and action can learn interpretable game 
strategies while maintaining neural network trainability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import random
import numpy as np
np.random.seed(7)
torch.manual_seed(7)

# Auto-detect device
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
except Exception:
    device = torch.device("cpu")
    print(f"Using device: {device} (fallback)")

class ReplayBuffer:
    """Standard DQN experience replay buffer"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def last_reward(self):
        """Get the reward from the most recently added experience"""
        if len(self.buffer) == 0:
            return 0.0
        return self.buffer[self.position - 1][2]

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class ConcreteLayer(nn.Module):
    """
    Layer 1: Concrete Pattern Detection
    
    PURPOSE: Detect low-level tactical patterns on the board
    INPUT: Raw board propositions [batch, 9, 2] 
    OUTPUT: Concrete features [batch, 12, 2]
    
    RULES: Pure template matching, no variables
    Each rule detects one specific type of pattern:
    - Empty corners, occupied center, opponent threats, etc.
    """
    
    def __init__(self, input_props=9, output_props=12):
        super(ConcreteLayer, self).__init__()
        self.input_props = input_props   # 9 board squares
        self.output_props = output_props # 12 concrete features
        self.L = 2  # proposition length [player, position]
        
        # Create 12 pattern detection rules
        self.rules = nn.ModuleList()
        for i in range(output_props):
            rule = nn.Module()
            
            # Template pattern to match against
            # Initialize with diverse patterns across the space
            template = torch.FloatTensor([
                random.choice([-1, 0, 1]),  # player: opponent/empty/self
                random.uniform(-1, 1)       # position: anywhere on board
            ])
            rule.template = nn.Parameter(template)
            
            # Cylindrification factors (0=strict constant, 1=flexible variable)
            # Start with mostly constant matching for concrete features
            γ = torch.FloatTensor([0.3, 0.3])  # Somewhat flexible
            rule.γs = nn.Parameter(γ)
            
            self.rules.append(rule)
    
    @staticmethod
    def sigmoid(x):
        """Clamp to [0,1] range"""
        return torch.clamp(x, 0.0, 1.0)
    
    def forward(self, board_props):
        """
        Detect concrete patterns in board state
        
        INPUT: board_props [batch, 9, 2] - raw board squares
        OUTPUT: concrete_features [batch, 12, 2] - detected patterns
        """
        batch_size = board_props.shape[0]
        concrete_features = torch.zeros(batch_size, self.output_props, self.L, device=board_props.device)
        
        for i, rule in enumerate(self.rules):
            # Find best matching square for this pattern
            match_scores = torch.zeros(batch_size, self.input_props, device=board_props.device)
            
            for l in range(self.L):  # For [player, position]
                γ = self.sigmoid(rule.γs[l])
                template_val = rule.template[l]
                board_vals = board_props[:, :, l]  # [batch, 9]
                
                # Match quality (lower = better match)
                diff = (template_val - board_vals) ** 2
                match_penalty = (1 - γ) * diff  # Only penalize if constant mode
                match_scores += match_penalty
            
            # Soft attention over squares (differentiable)
            attention_weights = F.softmax(-match_scores, dim=1)  # [batch, 9]
            
            # Weighted average of matched squares
            matched_square = torch.einsum('bi,bil->bl', attention_weights, board_props)  # [batch, 2]
            
            # Output concrete feature: [pattern_strength, spatial_focus]
            pattern_strength = torch.exp(-match_scores.min(dim=1).values)  # [batch]
            spatial_focus = matched_square[:, 1]  # Use position component
            
            # Clamp outputs to prevent explosion in next layer
            concrete_features[:, i, 0] = torch.clamp(pattern_strength, 0.0, 1.0)
            concrete_features[:, i, 1] = torch.clamp(spatial_focus, -1.0, 1.0)
        
        return concrete_features

class AbstractionLayer(nn.Module):
    """
    Layer 2: Strategic Abstraction 
    
    PURPOSE: Combine concrete features into strategic concepts
    INPUT: Concrete features [batch, 12, 2]
    OUTPUT: Strategic concepts [batch, 6, 2]
    
    RULES: Complex reasoning with variables for feature binding
    Each rule creates one strategic concept by combining concrete features:
    - Fork opportunities, defensive needs, positional advantages, etc.
    """
    
    def __init__(self, input_props=12, output_props=6, variables=4):
        super(AbstractionLayer, self).__init__()
        self.input_props = input_props   # 12 concrete features
        self.output_props = output_props # 6 strategic concepts
        self.variables = variables       # 4 variable slots per rule
        self.L = 2  # proposition length
        self.J = 2  # premises per rule (can combine 2 concrete features)
        
        # Create 6 strategic reasoning rules
        self.rules = nn.ModuleList()
        for i in range(output_props):
            rule = nn.Module()
            
            # Each rule has 2 premises for combining concrete features
            rule.body = nn.ModuleList()
            for j in range(self.J):
                body_layer = nn.Linear(self.L, self.variables)
                # Initialize body weights smaller
                nn.init.xavier_uniform_(body_layer.weight, gain=0.1)
                nn.init.constant_(body_layer.bias, 0.0)
                rule.body.append(body_layer)
            
            # Head network: variables → strategic concept [value, preference]
            rule.head = nn.Linear(self.variables, self.L)
            # Initialize head weights smaller
            nn.init.xavier_uniform_(rule.head.weight, gain=0.1)
            nn.init.constant_(rule.head.bias, 0.0)
            
            # Templates for matching concrete features (smaller initial values)
            templates = torch.FloatTensor(self.J, self.L).uniform_(-0.5, 0.5)
            rule.templates = nn.Parameter(templates)
            
            # Cylindrification factors
            γs = torch.FloatTensor(self.J, self.L).uniform_(0.4, 0.8)  # More flexible
            rule.γs = nn.Parameter(γs)
            
            self.rules.append(rule)
    
    @staticmethod
    def sigmoid(x):
        return torch.clamp(x, 0.0, 1.0)
    
    def forward(self, concrete_features):
        """
        Form strategic concepts from concrete features
        
        INPUT: concrete_features [batch, 12, 2]
        OUTPUT: strategic_concepts [batch, 6, 2]
        """
        batch_size = concrete_features.shape[0]
        strategic_concepts = torch.zeros(batch_size, self.output_props, self.L, device=concrete_features.device)
        
        for i, rule in enumerate(self.rules):
            captured_vars = torch.zeros(batch_size, self.variables, device=concrete_features.device)
            
            # Each premise matches against concrete features and captures variables
            for j in range(self.J):
                # Match this premise against all concrete features
                match_scores = torch.zeros(batch_size, self.input_props, device=concrete_features.device)
                
                for l in range(self.L):
                    γ = self.sigmoid(rule.γs[j, l])
                    template = rule.templates[j, l]
                    feature_vals = concrete_features[:, :, l]  # [batch, 12]
                    
                    diff = (template - feature_vals) ** 2
                    match_penalty = (1 - γ) * diff
                    match_scores += match_penalty
                
                # Soft selection of best matching concrete feature
                attention_weights = F.softmax(-match_scores, dim=1)  # [batch, 12]
                selected_feature = torch.einsum('bi,bil->bl', attention_weights, concrete_features)  # [batch, 2]
                
                # Capture variables from selected feature
                vars_captured = rule.body[j](selected_feature)  # [batch, variables]
                captured_vars += vars_captured
            
            # Generate strategic concept from captured variables
            strategic_concept = rule.head(captured_vars)  # [batch, 2]
            # Clamp strategic concepts to prevent explosion
            strategic_concept = torch.clamp(strategic_concept, -5.0, 5.0)
            strategic_concepts[:, i, :] = strategic_concept
        
        return strategic_concepts

class ActionLayer(nn.Module):
    """
    Layer 3: Action Policy Generation
    
    PURPOSE: Map strategic concepts directly to action Q-values  
    INPUT: Strategic concepts [batch, 6, 2]
    OUTPUT: Q-values [batch, 9]
    
    RULES: Direct linear mapping, no variables
    Each rule corresponds to one board square and computes its Q-value
    based on how well the strategic concepts support that action.
    """
    
    def __init__(self, input_props=6, output_actions=9):
        super(ActionLayer, self).__init__()
        self.input_props = input_props     # 6 strategic concepts
        self.output_actions = output_actions # 9 board squares
        self.L = 2  # proposition length
        
        # Create 9 action evaluation rules (one per square)
        self.action_rules = nn.ModuleList()
        for square in range(output_actions):
            # Each rule weights strategic concepts to produce Q-value for this square
            rule = nn.Linear(input_props * self.L, 1)  # 6*2=12 inputs → 1 Q-value
            # Initialize with smaller weights to prevent early explosion
            nn.init.xavier_uniform_(rule.weight, gain=0.1)
            nn.init.constant_(rule.bias, 0.0)
            self.action_rules.append(rule)
    
    def forward(self, strategic_concepts):
        """
        Generate Q-values from strategic concepts
        
        INPUT: strategic_concepts [batch, 6, 2]  
        OUTPUT: q_values [batch, 9]
        """
        batch_size = strategic_concepts.shape[0]
        
        # Flatten strategic concepts for linear layers
        strategic_flat = strategic_concepts.view(batch_size, -1)  # [batch, 12]
        
        # Each action rule produces Q-value for one square
        q_values = torch.zeros(batch_size, self.output_actions, device=strategic_concepts.device)
        
        for square, rule in enumerate(self.action_rules):
            q_values[:, square] = rule(strategic_flat).squeeze(-1)  # [batch]
        
        return q_values

class LogicNetwork(nn.Module):
    """
    3-Layer Hierarchical Logic Network
    
    ARCHITECTURE:
    Input [9×2] → ConcreteLayer → [12×2] → AbstractionLayer → [6×2] → ActionLayer → [9×1]
    
    GRADIENT FLOW:
    Loss → Q-values → ActionLayer → AbstractionLayer → ConcreteLayer → Input
    All layers are differentiable, enabling end-to-end learning of logical rules.
    """
    
    def __init__(self, input_dim=18, action_dim=9):
        super(LogicNetwork, self).__init__()
        
        # Reshape parameters
        self.W = 9  # board squares
        self.L = 2  # proposition length [player, position]
        
        # Three-layer hierarchy
        self.concrete = ConcreteLayer(input_props=9, output_props=12)
        self.abstraction = AbstractionLayer(input_props=12, output_props=6, variables=4)
        self.action = ActionLayer(input_props=6, output_actions=9)
    
    def forward(self, state):
        """
        Hierarchical logical reasoning
        
        INPUT: state [batch, 18] - flattened board
        OUTPUT: q_values [batch, 9] - action Q-values
        """
        batch_size = state.shape[0]
        
        # Reshape to proposition format
        board_props = state.reshape(batch_size, self.W, self.L)  # [batch, 9, 2]
        
        # Layer 1: Detect concrete patterns
        concrete_features = self.concrete(board_props)  # [batch, 12, 2]
        
        # Layer 2: Form strategic concepts
        strategic_concepts = self.abstraction(concrete_features)  # [batch, 6, 2]
        
        # Layer 3: Generate action Q-values
        q_values = self.action(strategic_concepts)  # [batch, 9]
        
        return q_values
    
    def choose_action(self, state, epsilon=0.1, deterministic=False):
        """Action selection with exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self(state_tensor).squeeze(0)  # [9]
        
        # Mask invalid actions (occupied squares)
        state_reshaped = state_tensor.reshape(self.W, self.L)
        valid_mask = (state_reshaped[:, 0] == 0)  # Empty squares
        
        if not valid_mask.any():
            return random.randint(0, 8)  # Fallback
        
        q_values[~valid_mask] = float('-inf')  # Mask invalid actions
        
        # Epsilon-greedy exploration
        if not deterministic and random.random() < epsilon:
            valid_actions = torch.where(valid_mask)[0]
            return valid_actions[random.randint(0, len(valid_actions)-1)].item()
        
        return torch.argmax(q_values).item()

class DQN:
    """
    Deep Q-Network with 3-Layer Logic Network
    
    LEARNING APPROACH:
    1. Concrete layer learns tactical pattern recognition
    2. Abstraction layer learns strategic concept formation  
    3. Action layer learns concept-to-action mapping
    
    All layers trained end-to-end via backpropagation from TD-error.
    """
    
    def __init__(self, action_dim=9, state_dim=18, learning_rate=3e-5, gamma=0.9):
        self.anet = LogicNetwork(state_dim, action_dim).to(device)
        self.tnet = LogicNetwork(state_dim, action_dim).to(device)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.gamma = gamma
        self.lr = learning_rate  # Add missing lr attribute
        self.optimizer = optim.Adam(self.anet.parameters(), lr=learning_rate, weight_decay=1e-4)  # Add weight decay
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Keep more exploration
        self.epsilon_decay = 0.9995  # Slower decay
        
        # Terminal state for TicTacToe
        self.endState = [
            0, -1.0,    # Square 0: empty, top-left
            0, -0.75,   # Square 1: empty, top-center
            0, -0.5,    # Square 2: empty, top-right
            0, -0.25,   # Square 3: empty, mid-left
            0,  0.0,    # Square 4: empty, center
            0,  0.25,   # Square 5: empty, mid-right
            0,  0.5,    # Square 6: empty, bottom-left
            0,  0.75,   # Square 7: empty, bottom-center
            0,  1.0     # Square 8: empty, bottom-right
        ]
    
    def choose_action(self, state, deterministic=False):
        return self.anet.choose_action(
            state, 
            epsilon=self.epsilon if not deterministic else 0.0,
            deterministic=deterministic
        )
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update(self, batch_size, reward_scale=1.0):
        if len(self.replay_buffer) < batch_size:
            return None
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Forward pass through 3-layer logic network
        q_values = self.anet(state)
        next_q_values = self.tnet(next_state)
        
        # Tighter clamping to prevent explosion
        q_values = torch.clamp(q_values, -5.0, 5.0)
        next_q_values = torch.clamp(next_q_values, -5.0, 5.0)
        
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        # Use Huber loss (more robust to outliers than MSE)
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach())
        
        # Backprop through all three layers with stronger gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.anet.parameters(), 0.5)  # Stronger clipping
        self.optimizer.step()
        
        return loss.item()
    
    def sync(self):
        """Sync target network"""
        self.tnet.load_state_dict(self.anet.state_dict())
    
    def save_net(self, filename):
        """Save network"""
        import os
        os.makedirs("PyTorch_models", exist_ok=True)
        filepath = f"PyTorch_models/{filename}.dict"
        torch.save(self.anet.state_dict(), filepath)
        print(f"3-Layer Logic model saved to {filepath}")
    
    def load_net(self, filename):
        """Load network"""
        filepath = f"PyTorch_models/{filename}.dict"
        self.anet.load_state_dict(torch.load(filepath))
        self.sync()
        print(f"3-Layer Logic model loaded from {filepath}")
    
    def net_info(self):
        """Network topology info"""
        total_params = sum(p.numel() for p in self.anet.parameters())
        topology = "concrete(9-12).abstract(12-6).action(6-9)"
        return (topology, total_params)
    
    def play_random(self, state, action_space):
        """Random baseline player"""
        empty_squares = []
        for i in range(9):
            if state[i * 2] == 0:  # Player index
                empty_squares.append(i)
        
        if not empty_squares:
            return random.randint(0, 8)
        
        return random.choice(empty_squares)
    
    def display_architecture(self):
        """Display learned logical rules"""
        print("\n" + "="*80)
        print("3-LAYER LOGIC NETWORK ARCHITECTURE")
        print("="*80)
        
        print(f"Layer 1 (Concrete): {self.anet.concrete.output_props} pattern detectors")
        print(f"Layer 2 (Abstraction): {self.anet.abstraction.output_props} strategic concepts") 
        print(f"Layer 3 (Action): {self.anet.action.output_actions} action evaluators")
        
        print(f"\nTotal parameters: {sum(p.numel() for p in self.anet.parameters())}")
        print("="*80 + "\n")
    
    def display_rules(self, sample_state=None):
        """Detailed examination of learned rules"""
        if sample_state is None:
            # Use empty board as sample
            sample_state = self.endState
        
        print("\n" + "="*100)
        print("DETAILED RULE ANALYSIS")
        print("="*100)
        
        # Convert sample state to tensor
        state_tensor = torch.FloatTensor(sample_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get activations at each layer
            board_props = state_tensor.reshape(1, 9, 2)
            concrete_features = self.anet.concrete(board_props)
            strategic_concepts = self.anet.abstraction(concrete_features)
            q_values = self.anet.action(strategic_concepts)
            
            print(f"\nSample state: {sample_state[:10]}...")  # Show first 10 values
            print(f"Q-values: {q_values.squeeze().cpu().numpy()}")            
            print(f"Q-value range: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")
            
            # LAYER 1 ANALYSIS: Concrete patterns
            print("\n" + "-"*50)
            print("LAYER 1: CONCRETE PATTERN DETECTORS")
            print("-"*50)
            
            for i, rule in enumerate(self.anet.concrete.rules):
                template = rule.template.detach().cpu().numpy()
                gammas = rule.γs.detach().cpu().numpy()
                
                # Compute activation for this rule on sample state
                activation = concrete_features[0, i, 0].item()  # Pattern strength
                focus = concrete_features[0, i, 1].item()       # Spatial focus
                
                print(f"  Rule {i+1:2d}: Template=[{template[0]:+.2f},{template[1]:+.2f}] "
                      f"γ=[{gammas[0]:.2f},{gammas[1]:.2f}] "
                      f"→ strength={activation:.3f}, focus={focus:.3f}")
                
                if abs(template[0]) > 2 or abs(template[1]) > 2:
                    print(f"    ⚠️  LARGE TEMPLATE VALUES detected!")
                if gammas[0] > 0.9 or gammas[1] > 0.9:
                    print(f"    📝 Variable mode (γ≈1)")
                elif gammas[0] < 0.1 and gammas[1] < 0.1:
                    print(f"    🔒 Constant mode (γ≈0)")
            
            # LAYER 2 ANALYSIS: Strategic concepts
            print("\n" + "-"*50)
            print("LAYER 2: STRATEGIC CONCEPT FORMATION")
            print("-"*50)
            
            for i, rule in enumerate(self.anet.abstraction.rules):
                concept_value = strategic_concepts[0, i, 0].item()
                concept_preference = strategic_concepts[0, i, 1].item()
                
                print(f"  Concept {i+1}: value={concept_value:+.3f}, preference={concept_preference:+.3f}")
                
                # Check rule parameters
                templates = rule.templates.detach().cpu().numpy()
                gammas = rule.γs.detach().cpu().numpy()
                head_weights = rule.head.weight.detach().cpu().numpy()
                head_bias = rule.head.bias.detach().cpu().numpy()
                
                print(f"    Templates: {templates.flatten()}")
                print(f"    Head weights norm: {np.linalg.norm(head_weights):.3f}")
                print(f"    Head bias: {head_bias}")
                
                if np.any(np.abs(templates) > 5):
                    print(f"    ⚠️  EXPLOSIVE TEMPLATES!")
                if np.linalg.norm(head_weights) > 10:
                    print(f"    ⚠️  LARGE HEAD WEIGHTS!")
            
            # LAYER 3 ANALYSIS: Action preferences
            print("\n" + "-"*50)
            print("LAYER 3: ACTION EVALUATORS")
            print("-"*50)
            
            for i, rule in enumerate(self.anet.action.action_rules):
                q_val = q_values[0, i].item()
                weights = rule.weight.detach().cpu().numpy().flatten()
                bias = rule.bias.detach().cpu().numpy()[0]
                
                print(f"  Square {i}: Q={q_val:+.3f} | bias={bias:+.3f} | weights_norm={np.linalg.norm(weights):.3f}")
                
                if abs(q_val) > 20:
                    print(f"    ⚠️  EXTREME Q-VALUE!")
                if np.linalg.norm(weights) > 5:
                    print(f"    ⚠️  LARGE WEIGHTS!")
                if abs(bias) > 5:
                    print(f"    ⚠️  LARGE BIAS!")
        
        print("\n" + "="*100)
        
    def check_gradient_flow(self):
        """Check if gradients are flowing properly"""
        print("\n" + "="*60)
        print("GRADIENT FLOW ANALYSIS")
        print("="*60)
        
        total_norm = 0
        param_count = 0
        
        for name, param in self.anet.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                param_count += 1
                
                if grad_norm > 1.0:
                    print(f"⚠️  {name}: grad_norm={grad_norm:.6f} (HIGH)")
                elif grad_norm > 0.1:
                    print(f"✓  {name}: grad_norm={grad_norm:.6f}")
                else:
                    print(f"📉 {name}: grad_norm={grad_norm:.6f} (low)")
            else:
                print(f"❌ {name}: NO GRADIENTS")
        
        total_norm = (total_norm) ** 0.5
        print(f"\nTotal gradient norm: {total_norm:.6f}")
        print(f"Parameters with gradients: {param_count}")
        print("="*60)

if __name__ == "__main__":
    # Quick test of the 3-layer architecture - runs when executed directly
    print("Testing 3-Layer Logic Network...")
    
    # Create network
    dqn = DQN()
    
    # Test with empty board
    empty_board = [0, -1.0, 0, -0.75, 0, -0.5, 0, -0.25, 0, 0.0, 
                   0, 0.25, 0, 0.5, 0, 0.75, 0, 1.0]
    
    action = dqn.choose_action(empty_board, deterministic=True)
    print(f"Chose action (square): {action}")
    
    # Display architecture
    dqn.display_architecture()
    
    # NEW: Debug the initial rules before training
    print("\\n" + "="*50)
    print("INITIAL RULE ANALYSIS")
    print("="*50)
    dqn.debug_rules(empty_board)
    
    print("3-Layer Logic Network test completed!")
    print("Ready for import as logic.hierarchical module")
    print("\\nTo debug rules during training, call:")
    print("  dqn.debug_rules()        # Examine all learned rules")
    print("  dqn.check_gradient_flow() # Check gradient issues")
