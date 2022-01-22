# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import test_snake
from composed_functions.dqn_helper import dqn, load_checkpoint, load_checkpoint_to_snake, save_checkpoint
from federated_learning.federated_average import agregate

if __name__ == "__main__":
    '''
    Trains multiple seperated DQN-agents in the same environment that they compete in.
    '''

    episode_amount: int = 1_000_000
    env_episode_amount: int = 1
    save_every: int = 50
    board_dim: int = 20
    state_size: int = 9
    model_type: str = "fed_classic_together"
    model_id: str = "_{}_{}x{}+{}".format(model_type, state_size, state_size, 4)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    dqn_snake_1: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4,
        action_size         = 4,
        init_snake_lengths  = array([2, 100]),
        seed                = 1337,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.0001, 50_000),
        learning_rate       = 2.5e-5,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 100_000
    )

    dqn_snake_2: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4,
        action_size         = 4,
        init_snake_lengths  = array([2, 100]),
        seed                = 1337,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.0001, 50_000),
        learning_rate       = 2.5e-5,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 100_000
    )

    #           * testing snake agent (small snake with 0 epsilon)
    dqn_snake_TEST: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(0., 0., 50_000),
        learning_rate       = 2.5e-5,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 100_000
    )

    #           * default board with fixed size and only one food
    board_food: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(epsilon=Epsilon(1, 0.0, 1_000))
        },
    )
    board_TEST: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile()
        },
    )

    checkpoint = load_checkpoint(model_path)
    load_checkpoint_to_snake(dqn_snake_1, checkpoint)
    load_checkpoint_to_snake(dqn_snake_2, checkpoint)

    for i in range(episode_amount):
        medians: list = []

        for snake, board in [(dqn_snake_1, board_mine), (dqn_snake_2, board_food)]:
            # Train each snake seperatly for env_episode_amount episodes
            medians.append(
                dqn(
                    board,
                    snake,
                    env_episode_amount,
                    lambda: observation_cat(
                        observation_near(
                            board=board,
                            snake=snake,
                            kernel=array([state_size, state_size])
                        ),
                        observation_to_bool(observation_food(snake))
                    )
                )
            )
        print('\rEpisode {}\tAverage Scores {}\tRandom act chance {:.6f}'.format(i * env_episode_amount, ["{:.2f}".format(median) for median in medians], dqn_snake_food.epsilon(dqn_snake_food.board.run)), end="")

        # Do a fedaverage between them
        agregate([dqn_snake_mine, dqn_snake_food], dqn_snake_TEST)

        # Save their model
        if i % save_every == 0:
            print('\rEpisode {}\tAverage Scores {}\tRandom act chance {:.6f}'.format(i * env_episode_amount, ["{:.2f}".format(median) for median in medians], dqn_snake_food.epsilon(dqn_snake_food.board.run)))
            save_checkpoint(dqn_snake_mine, model_path)

        if (i % 250 == 0):
            test_snake(
                board=board_combined,
                snake=dqn_snake_TEST,
                observation_function = lambda: observation_cat(
                    observation_near(
                        board=board_combined,
                        snake=dqn_snake_TEST,
                        kernel=array([state_size, state_size])
                    ),
                    observation_to_bool(observation_food(dqn_snake_TEST))
                )
            )