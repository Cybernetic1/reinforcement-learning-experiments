from gym.envs.registration import register

# Env registration
# ==========================

register(
    id='TicTacToe-plain-v0',		# plain version
    entry_point='gym_tictactoe.tic_tac_toe_plain:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-logic-v0',		# logic version 0
    entry_point='gym_tictactoe.tic_tac_toe_logic:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-logic-v1',		# logic version 1
    entry_point='gym_tictactoe.tic_tac_toe_logic1:TicTacToeEnv',
    reward_threshold=1000
)
