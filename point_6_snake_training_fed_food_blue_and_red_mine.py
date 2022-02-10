# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import test_snake
from composed_functions.dqn_helper import dqn, load_checkpoint, load_checkpoint_to_snake, save_checkpoint
from federated_learning.federated_average import agregate

if __name__ == "__main__":
    '''
    Trains two seperated DQN-agents in the seperate environment with advanced rules (blue or red mine and fruit)
    Does a fedaverage between the agents to show that multiple agents can learn from one and another and achive the same preformance as normal
    '''

    episode_amount: int = 1_000_000
    env_episode_amount: int = 1
    save_every: int = 50
    board_dim: int = 20
    state_size: int = 5
    model_type: str = "fed_food_blue_and_red_mine"
    model_id: str = "_{}_{}x{}+{}".format(model_type, state_size, state_size, 4)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    # Snake 1 and its environment
    dqn_snake_1: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.0001, 45_000),
        learning_rate       = 1e-3,
        tau                 = 1e-3,
        update_every        = 256,
        buffer_size         = 100_000
    )
    board_1: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 10000,
        snakes                  = [dqn_snake_1],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(epsilon=Epsilon(1., 0., 40_000)),
            "mine_tile": MineTile(epsilon=Epsilon(1., 0., 40_000))
        },
    )

    # Snake and its environment with food
    dqn_snake_2: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.0001, 45_000),
        learning_rate       = 1e-3,
        tau                 = 1e-3,
        update_every        = 256,
        buffer_size         = 100_000
    )
    board_2: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 10000,
        snakes                  = [dqn_snake_2],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(epsilon=Epsilon(1., 0., 40_000)),
            "mine_tile": MineTile(epsilon=Epsilon(1., 0., 40_000),visual=6)
        },
    )

    dqn_snake_TEST: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(0, 0.000, 45_000),
        learning_rate       = 1e-3,
        tau                 = 1e-3,
        update_every        = 256,
        buffer_size         = 100_000
    )
    # Test snake and its environment with both food and mines
    board_TEST: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 5,
        snakes                  = [dqn_snake_TEST],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(),
            "mine_tile": MineTile(visual=6),
            "_mine_tile": MineTile()
        },
    )
    checkpoint = load_checkpoint(model_path)
    inital_episode: int = load_checkpoint_to_snake(dqn_snake_1, checkpoint) + 1
    board_1.run = load_checkpoint_to_snake(dqn_snake_2, checkpoint)
    board_2.run = board_1.run

    for i in range(inital_episode, episode_amount):
        medians: list = []

        for snake, board in [(dqn_snake_1, board_1), (dqn_snake_2, board_2)]:
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
        print('\rEpisode {}\tAverage Scores {}\tRandom act chance {:.6f}'.format(i * env_episode_amount, ["{:.2f}".format(median) for median in medians], dqn_snake_1.epsilon(dqn_snake_1.board.run)), end="")

        # Do a fedaverage between them
        agregate([dqn_snake_1, dqn_snake_2], dqn_snake_TEST)

        # Save their model
        if i % save_every == 0:
            print('\rEpisode {}\tAverage Scores {}\tRandom act chance {:.6f}'.format(i * env_episode_amount, ["{:.2f}".format(median) for median in medians], dqn_snake_1.epsilon(dqn_snake_1.board.run)))
            save_checkpoint(dqn_snake_1, i, board_TEST, model_path)

        if (i % 250 == 0):
            test_snake(
                board=board_TEST,
                snake=dqn_snake_TEST,
                observation_function = lambda: observation_cat(
                    observation_near(
                        board=board_TEST,
                        snake=dqn_snake_TEST,
                        kernel=array([state_size, state_size])
                    ),
                    observation_to_bool(observation_food(dqn_snake_TEST))
                )
            )
        if board_1.replay_interval != 0 and i % board_1.replay_interval == 0:
            save_checkpoint(dqn_snake_1, i, board_TEST, "replays/replay{}/replay{}_episode_{}.pth".format(model_id, model_id, i))