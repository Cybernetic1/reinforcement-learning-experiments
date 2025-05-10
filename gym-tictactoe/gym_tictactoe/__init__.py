from gym.envs.registration import register

# Env registration
# ==========================

register(
    id='TicTacToe-plain-v0',		# plain version
    entry_point='gym_tictactoe.TTT_plain:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-logic-v0',		# logic version
    entry_point='gym_tictactoe.TTT_logic:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-logic-dim1-v0',	# logic dim1
    entry_point='gym_tictactoe.TTT_logic_dim1:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-logic-dim2-v1',	# logic dim2
    entry_point='gym_tictactoe.TTT_logic_dim2:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-logic-dim2-v2',	# logic dim2 uniform
    entry_point='gym_tictactoe.TTT_logic_dim2_uniform:TicTacToeEnv',
    reward_threshold=1000
)


register(
    id='TicTacToe-logic-dim2-v3',	# logic dim2 polar
    entry_point='gym_tictactoe.TTT_logic_dim2_polar:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-logic-dim2-intermediate-v0',	# logic dim2, with intermediate memory
    entry_point='gym_tictactoe.TTT_logic_dim2_intermediate:TicTacToeEnv',
    reward_threshold=1000
)
