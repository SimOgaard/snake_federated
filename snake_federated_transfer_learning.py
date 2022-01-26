# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import test_snake
from composed_functions.dqn_helper import dqn, load_checkpoint, load_checkpoint_to_snake, save_checkpoint
from federated_learning.federated_average import agregate

if __name__ == "__main__":
    '''
    Trains two seperated DQN-agents in two seperate environments with same rules (mine and fruit)
    Does a fedaverage between the two agents to show that two agents in seperate environments;
        can transfer what they have learned to eachother even if the other one has no experience of it
    '''

    episode_amount: int = 1_000_000
    env_episode_amount: int = 1
    save_every: int = 50
    board_dim: int = 20
    state_size: int = 5
    model_type: str = "fed_transfer"
    model_id: str = "_{}_{}x{}+{}".format(model_type, state_size, state_size, 4)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    # Snake and its environment with only mines
    dqn_snake_mine: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.0001, 50_000),
        learning_rate       = 2.5e-5,
        tau                 = 1e-3,
        update_every        = 256,
        buffer_size         = 100_000
    )
    board_mine: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 1000,
        snakes                  = [dqn_snake_mine],
        tiles_populated         = {
            "air_tile": AirTile(reward=0.01),
            "wall_tile": WallTile(),
            "mine_tile": MineTile(epsilon=Epsilon(1., 0., 10_000))
        },
    )

    # Snake and its environment with food
    dqn_snake_food: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.0001, 50_000),
        learning_rate       = 2.5e-5,
        tau                 = 1e-3,
        update_every        = 256,
        buffer_size         = 100_000
    )
    board_food: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_food],
        tiles_populated         = {
            "air_tile": AirTile(reward=0.01),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(epsilon=Epsilon(1., 0., 10_000))
        },
    )

    dqn_snake_TEST: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(0, 0.000, 50_000),
        learning_rate       = 2.5e-5,
        tau                 = 1e-3,
        update_every        = 256,
        buffer_size         = 100_000
    )
    # Test snake and its environment with both food and mines
    board_combined: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 5,
        snakes                  = [dqn_snake_TEST],
        tiles_populated         = {
            "air_tile": AirTile(reward=0.01),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(),
            "mine_tile": MineTile()
        },
    )
    checkpoint = load_checkpoint(model_path)
    initial_episode: int = load_checkpoint_to_snake(dqn_snake_mine, checkpoint) + 1
    board_food.run = load_checkpoint_to_snake(dqn_snake_food, checkpoint)
    board_mine.run = board_food.run

    for i in range(initial_episode, episode_amount):
        medians: list = []

        for snake, board in [(dqn_snake_mine, board_mine), (dqn_snake_food, board_food)]:
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
            save_checkpoint(dqn_snake_mine, i, board_combined, model_path)

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
        if board_mine.replay_interval != 0 and i % board_mine.replay_interval == 0:
            save_checkpoint(dqn_snake_mine, i, board_combined, "replays/replay{}/replay{}_episode_{}.pth".format(model_id, model_id, i))