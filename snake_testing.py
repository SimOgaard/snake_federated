# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import test_snake
from composed_functions.dqn_helper import load_checkpoint, load_checkpoint_to_snake

if __name__ == "__main__":
    '''
    Tests saved specified model
    '''

    episode_amount: int = 1_000_000
    board_dim: int = 20
    state_size: int = 5
    model_type: str = "fed_none"
    model_id: str = "_{}_{}x{}+{}".format(model_type, state_size, state_size, 4)
    #model_path: str = 'models/checkpoint{}.pth'.format(model_id)
    model_path: str = 'models/checkpoint_fed_none_5x5+4.pth'

    dqn_snake: DQNAgent = DQNAgent(
        state_size    = state_size**2 + 4,
        action_size   = 4,
        init_snake_lengths=array([2, 2]),
        seed          = 1337,
        batch_size    = 128,
        gamma         = 0.999,
        epsilon       = Epsilon(0, 0, 1_000),
        learning_rate = 1e-3,
        tau           = 1e-3,
        update_every  = 32,
        buffer_size   = 100_000
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

    test_snake(
        board=board,
        snake=dqn_snake,
        observation_function = lambda: observation_cat(
            observation_near(
                board=board,
                snake=dqn_snake,
                kernel=array([state_size, state_size])
            ),
            observation_to_bool(observation_food(dqn_snake))
        ),
        test_amount=1000000,
        visualize=True
    )