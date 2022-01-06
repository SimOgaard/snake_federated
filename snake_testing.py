# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import pretty_print, display, display_run
from composed_functions.dqn_helper import load_checkpoint, load_checkpoint_to_snake

if __name__ == "__main__":
    '''
    Tests saved model
    '''

    episode_amount: int = 100_000
    board_dim: int = 5
    state_size: int = 5
    model_id: str = "{}x{}".format(state_size, state_size)
    model_path: str = 'snake_rl/models/checkpoint{}.pth'.format(model_id)

    dqn_snake: DQNAgent = DQNAgent(
        state_size    = state_size**2,
        action_size   = 4,
        init_snake_lengths=array([2, 2]),
        seed          = 1337,
        batch_size    = 128,
        gamma         = 0.999,
        epsilon       = Epsilon(0, 0., 100_000),
        learning_rate = 5e-4,
        tau           = 1e-3,
        update_every  = 32,
        buffer_size   = 500_000
    )

    board: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile()
        },
    )

    checkpoint = load_checkpoint(model_path)
    load_checkpoint_to_snake(dqn_snake, checkpoint)

    while board.run < episode_amount:
        display_run(board, dqn_snake, array([state_size, state_size]), pretty_print, observation_near)