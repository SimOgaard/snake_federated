# Repo imports
from cgi import test
from composed_functions.ui_helper import Replay_Snake
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.dqn_helper import load_checkpoint, load_checkpoint_to_snake

if __name__ == "__main__":
    '''
    Tests specified model in model_path on specified board
    '''

    episode_amount: int = 1_000_000
    board_dim: int = 20
    state_size: int = 5
    model_type: str = "fed_none"
    model_id: str = "_{}_{}x{}+{}".format(model_type, state_size, state_size, 4)
    #model_path: str = 'models/checkpoint{}.pth'.format(model_id)
    model_path: str = 'models\checkpoint_fed_late_food_blue_and_red_mine_5x5+4.pth'

    dqn_snake: DQNAgent = DQNAgent(
        state_size    = state_size**2 + 4,
        action_size   = 4,
        init_snake_lengths=array([2, 2]),
        seed          = 1337,
        batch_size    = 128,
        gamma         = 0.999,
        epsilon       = Epsilon(0, 0, 1_000),
        learning_rate = 5e-4,
        tau           = 5e-4,
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
            "food_tile": FoodTile(),
            "mine_tile": MineTile(),
            "_mine_tile": MineTile(visual=6),
        },
    )

    checkpoint = load_checkpoint(model_path)
    load_checkpoint_to_snake(dqn_snake, checkpoint)

    test_snake: Replay_Snake = Replay_Snake(
        board=board,
        snake=dqn_snake,
        kernel=state_size,
        observation_function = lambda: observation_cat(
            observation_near(
                board=board,
                snake=dqn_snake,
                kernel=array([state_size, state_size])
            ),
            observation_to_bool(observation_food(dqn_snake))
        ),
        test_amount=1000000,
        save_amount=1000
    )

    test_snake.show_replay_snake()