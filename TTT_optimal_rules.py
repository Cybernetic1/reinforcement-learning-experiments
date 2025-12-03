#!/usr/bin/env python3
"""
Hand-crafted optimal logic rules for Tic-Tac-Toe

This serves as a reference for what a symbolic rule-based system should learn.
Rules are prioritized from most to least important.

Board positions:
  0 1 2     TL TM TR
  3 4 5  =  ML MM MR
  6 7 8     BL BM BR

Player encoding: X=+1, O=-1, empty=0
"""

class TicTacToeRules:
    """
    Optimal Tic-Tac-Toe strategy as explicit logic rules.
    """
    
    # Win lines for checking patterns
    LINES = [
        [0, 1, 2],  # Top row
        [3, 4, 5],  # Middle row
        [6, 7, 8],  # Bottom row
        [0, 3, 6],  # Left column
        [1, 4, 7],  # Middle column
        [2, 5, 8],  # Right column
        [0, 4, 8],  # Diagonal \
        [2, 4, 6],  # Diagonal /
    ]
    
    CORNERS = [0, 2, 6, 8]
    CENTER = 4
    EDGES = [1, 3, 5, 7]
    
    @staticmethod
    def get_action(board, player):
        """
        Get best action for given board state.
        
        Args:
            board: list of 9 values (1=X, -1=O, 0=empty)
            player: 1 for X, -1 for O
        
        Returns:
            position (0-8) to play
        """
        opponent = -player
        
        # Priority 1: Win immediately if possible
        # Rule: IF two of mine in a line AND third empty THEN play third
        move = TicTacToeRules._find_winning_move(board, player)
        if move is not None:
            return move
        
        # Priority 2: Block opponent's winning move
        # Rule: IF two of opponent's in a line AND third empty THEN block third
        move = TicTacToeRules._find_winning_move(board, opponent)
        if move is not None:
            return move
        
        # Priority 3: Create fork (two winning threats)
        # Rule: IF can create two simultaneous winning threats THEN do it
        move = TicTacToeRules._find_fork(board, player)
        if move is not None:
            return move
        
        # Priority 4: Block opponent's fork
        # Rule: IF opponent can fork THEN block or force defense
        move = TicTacToeRules._block_fork(board, player, opponent)
        if move is not None:
            return move
        
        # Priority 5: Take center if available
        # Rule: IF center empty THEN play center
        if board[TicTacToeRules.CENTER] == 0:
            return TicTacToeRules.CENTER
        
        # Priority 6: Take opposite corner if opponent in corner
        # Rule: IF opponent in corner AND opposite corner empty THEN play opposite
        move = TicTacToeRules._opposite_corner(board, opponent)
        if move is not None:
            return move
        
        # Priority 7: Take any corner
        # Rule: IF any corner empty THEN play corner
        for corner in TicTacToeRules.CORNERS:
            if board[corner] == 0:
                return corner
        
        # Priority 8: Take any edge
        # Rule: IF any edge empty THEN play edge
        for edge in TicTacToeRules.EDGES:
            if board[edge] == 0:
                return edge
        
        # No moves available (shouldn't happen in valid game)
        return None
    
    @staticmethod
    def _find_winning_move(board, player):
        """Find a move that wins immediately."""
        for line in TicTacToeRules.LINES:
            count = sum(1 for pos in line if board[pos] == player)
            empty = [pos for pos in line if board[pos] == 0]
            
            # Two of player's pieces and one empty = winning move
            if count == 2 and len(empty) == 1:
                return empty[0]
        
        return None
    
    @staticmethod
    def _find_fork(board, player):
        """
        Find a move that creates a fork (two winning threats).
        A fork means after this move, there are 2+ ways to win on next turn.
        """
        for pos in range(9):
            if board[pos] != 0:
                continue
            
            # Try this move
            board[pos] = player
            
            # Count how many lines have 2 of player's pieces
            winning_lines = 0
            for line in TicTacToeRules.LINES:
                count = sum(1 for p in line if board[p] == player)
                empty = sum(1 for p in line if board[p] == 0)
                if count == 2 and empty == 1:
                    winning_lines += 1
            
            # Undo move
            board[pos] = 0
            
            # Fork = 2+ winning lines
            if winning_lines >= 2:
                return pos
        
        return None
    
    @staticmethod
    def _block_fork(board, player, opponent):
        """
        Block opponent's fork opportunity.
        If opponent can fork, either block it or force them to defend.
        """
        # Find all opponent's fork moves
        fork_moves = []
        for pos in range(9):
            if board[pos] != 0:
                continue
            
            # Try this move for opponent
            board[pos] = opponent
            
            # Count potential winning lines
            winning_lines = 0
            for line in TicTacToeRules.LINES:
                count = sum(1 for p in line if board[p] == opponent)
                empty = sum(1 for p in line if board[p] == 0)
                if count == 2 and empty == 1:
                    winning_lines += 1
            
            # Undo move
            board[pos] = 0
            
            if winning_lines >= 2:
                fork_moves.append(pos)
        
        if not fork_moves:
            return None
        
        # Strategy 1: Create a two-in-a-row to force opponent to defend
        # This prevents them from forking
        for pos in range(9):
            if board[pos] != 0:
                continue
            
            board[pos] = player
            
            # Check if this creates a threat (2 in a row)
            creates_threat = False
            for line in TicTacToeRules.LINES:
                count = sum(1 for p in line if board[p] == player)
                empty_in_line = [p for p in line if board[p] == 0]
                
                if count == 2 and len(empty_in_line) == 1:
                    # This creates a threat. Check if the empty square
                    # is not one of opponent's fork moves
                    if empty_in_line[0] not in fork_moves:
                        creates_threat = True
                        break
            
            board[pos] = 0
            
            if creates_threat:
                return pos
        
        # Strategy 2: If can't force defense, just block one fork move
        return fork_moves[0]
    
    @staticmethod
    def _opposite_corner(board, opponent):
        """Take opposite corner if opponent is in a corner."""
        corner_pairs = [(0, 8), (2, 6)]
        
        for c1, c2 in corner_pairs:
            if board[c1] == opponent and board[c2] == 0:
                return c2
            if board[c2] == opponent and board[c1] == 0:
                return c1
        
        return None
    
    @staticmethod
    def to_logic_notation():
        """
        Express the rules in a more formal logic notation.
        This shows what an ideal learned system should capture.
        """
        rules = """
OPTIMAL TIC-TAC-TOE RULES (Priority Order)
==========================================

Rule 1: WIN
∀ line ∈ Lines: 
  IF (count(my_pieces, line) = 2) ∧ (count(empty, line) = 1)
  THEN play(empty_position_in_line)
  CONFIDENCE: 1.0 (highest priority)

Rule 2: BLOCK
∀ line ∈ Lines:
  IF (count(opponent_pieces, line) = 2) ∧ (count(empty, line) = 1)
  THEN play(empty_position_in_line)
  CONFIDENCE: 0.9

Rule 3: FORK (create two threats)
∃ position p:
  IF after_playing(p) ⟹ (count(lines_with_2_mine) ≥ 2)
  THEN play(p)
  CONFIDENCE: 0.8

Rule 4: BLOCK_FORK
∃ position p where opponent_can_fork(p):
  IF ∃ move m: creates_forcing_threat(m) ∧ ¬enables_fork(m)
  THEN play(m)
  ELSE play(p)
  CONFIDENCE: 0.7

Rule 5: CENTER
IF board[center] = empty
THEN play(center)
CONFIDENCE: 0.6

Rule 6: OPPOSITE_CORNER
∀ corner_pair (c1, c2) ∈ {(TL,BR), (TR,BL)}:
  IF board[c1] = opponent ∧ board[c2] = empty
  THEN play(c2)
  CONFIDENCE: 0.5

Rule 7: ANY_CORNER
∃ c ∈ {TL, TR, BL, BR}:
  IF board[c] = empty
  THEN play(c)
  CONFIDENCE: 0.4

Rule 8: ANY_EDGE
∃ e ∈ {TM, ML, MR, BM}:
  IF board[e] = empty
  THEN play(e)
  CONFIDENCE: 0.3

PATTERN VOCABULARY NEEDED
=========================
To implement these rules, the system needs to recognize:

1. Line patterns:
   - "Two mine + one empty" → (X, X, 0) or permutations
   - "Two opponent + one empty" → (O, O, 0) or permutations
   
2. Position types:
   - Corner positions: {0, 2, 6, 8}
   - Center position: {4}
   - Edge positions: {1, 3, 5, 7}
   
3. Structural patterns:
   - Lines (8 total: 3 rows, 3 cols, 2 diagonals)
   - Corner pairs: (0,8) and (2,6)
   
4. Counting operators:
   - Count pieces in a line
   - Count potential winning lines

5. Simulation/lookahead:
   - "After playing position p, how many winning lines exist?"
   - Used for fork detection
"""
        return rules


def test_rules():
    """Test the rule system on some positions."""
    
    print("Testing Hand-Crafted Tic-Tac-Toe Rules")
    print("=" * 60)
    
    # Test case 1: Immediate win
    board = [1, 1, 0,   # X X _
             0, -1, 0,  # _ O _
             0, -1, 0]  # _ O _
    move = TicTacToeRules.get_action(board, player=1)
    print(f"\nTest 1 - Immediate win available:")
    print(f"Board: {board}")
    print(f"X should play: {move} (expected: 2 = TR)")
    assert move == 2
    
    # Test case 2: Block opponent win
    board = [1, 0, 0,   # X _ _
             0, -1, 0,  # _ O _
             0, -1, 0]  # _ O _
    move = TicTacToeRules.get_action(board, player=1)
    print(f"\nTest 2 - Block opponent:")
    print(f"Board: {board}")
    print(f"X should play: {move} (expected: 1 = TM, blocking column)")
    # Actually should block at position 1 (top middle) - the O column
    
    # Test case 3: Take center if empty
    board = [0, 0, 0,
             0, 0, 0,
             0, 0, 0]
    move = TicTacToeRules.get_action(board, player=1)
    print(f"\nTest 3 - Empty board:")
    print(f"X should play: {move} (expected: 4 = MM, center)")
    assert move == 4
    
    # Test case 4: Opposite corner
    board = [-1, 0, 0,  # O _ _
             0, 1, 0,   # _ X _
             0, 0, 0]   # _ _ _
    move = TicTacToeRules.get_action(board, player=1)
    print(f"\nTest 4 - Opposite corner:")
    print(f"Board: {board}")
    print(f"X should play: {move} (expected: 8 = BR, opposite corner)")
    assert move == 8
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    
    print("\n" + TicTacToeRules.to_logic_notation())


if __name__ == "__main__":
    test_rules()
