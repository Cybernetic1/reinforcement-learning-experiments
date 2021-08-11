from gym.envs.registration import register

# Env registration
# ==========================

# **** Note: gym forbids to name ID's other than ending with -v[0-9]

register(
    id='TicTacToe-v0',		# plain version
    entry_point='gym_tictactoe.tic_tac_toe_plain:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-v1',		# logic version
    entry_point='gym_tictactoe.tic_tac_toe_logic:TicTacToeEnv',
    reward_threshold=1000
)
