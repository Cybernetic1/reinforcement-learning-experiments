from gym.envs.registration import register

# Env registration
# ==========================

register(
    id='TicTacToe-plain',
    entry_point='gym_tictactoe.tic_tac_toe_plain:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-logic',
    entry_point='gym_tictactoe.tic_tac_toe_logic:TicTacToeEnv',
    reward_threshold=1000
)
