# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import pretty_print, display, display_run, test_snake
from composed_functions.dqn_helper import dqn, load_checkpoint, load_checkpoint_to_snake, save_checkpoint
from federated_learning.federated_average import agregate

if __name__ == "__main__":
    '''
    Trains multiple seperated DQN-agents in seperate environments with different rules
    Does a fedaverage between the agents to show that multiple rules for multiple agents can increase preformance/exploration
    '''

    episode_amount: int = 1_000_000
    env_episode_amount: int = 1
    save_every: int = 50
    board_dim: int = 20
    state_size: int = 9
    model_type: str = "fed_classic"
    model_id: str = "_{}_{}x{}+{}".format(model_type, state_size, state_size, 4)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    # snake for:
    #           * early game exploration (small snake length and high epsilon)
    #           * late game exploration (large snake and high epsilon)
    dqn_snake_exploration: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4,
        action_size         = 4,
        init_snake_lengths  = array([2, 100]),
        seed                = 1337,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.25, 50_000),
        learning_rate       = 1e-3,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 100_000
    )

    #           * early game exploitation (small snake length and low epsilon)
    #           * late game exploitation (large snake and low epsilon)
    dqn_snake_exploitation: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4,
        action_size         = 4,
        init_snake_lengths  = array([2, 100]),
        seed                = 1337,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(0.25, 0.0001, 50_000),
        learning_rate       = 1e-3,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 100_000
    )

    #           * normal snake agent (small snake with decreesing epsilon)
    dqn_snake_normal: DQNAgent = DQNAgent(
        state_size          = state_size**2 + 4,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.0001, 50_000),
        learning_rate       = 1e-3,
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
        learning_rate       = 1e-3,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 100_000
    )

    #           * board with static size and lots of food
    board_LOTS_of_food: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(epsilon=Epsilon(1, 0.25, 1_000))
        },
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
            "food_tile": FoodTile()
        },
    )

    # Lists holding all snakes and boards for easy itteration and mixing
    snakes: list = [dqn_snake_exploration, dqn_snake_exploitation, dqn_snake_normal]
    boards: list = [board_LOTS_of_food, board_food]

    checkpoint = load_checkpoint(model_path)
    load_checkpoint_to_snake(dqn_snake_normal, checkpoint)
    load_checkpoint_to_snake(dqn_snake_exploitation, checkpoint)
    load_checkpoint_to_snake(dqn_snake_exploration, checkpoint)

    for i in range(episode_amount):
        # train all snakes on all boards for env_episode_amount episodes
        #median: float = 0 # median length of dqn_snake_normal in board_food_mine
        medians: list = []
        for snake in snakes:
            for board in boards:
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
                
        print('\rEpisode {}\tAverage Scores {}\tRandom act chance {:.6f}'.format(i * env_episode_amount, ["{:.2f}".format(median) for median in medians], dqn_snake_normal.epsilon(dqn_snake_normal.board.run)), end="")

        # do a fedaverage between them
        agregate(snakes, dqn_snake_TEST)

        # Save their model
        if i % save_every == 0:
            print('\rEpisode {}\tAverage Scores {}\tRandom act chance {:.6f}'.format(i * env_episode_amount, ["{:.2f}".format(median) for median in medians], dqn_snake_normal.epsilon(dqn_snake_normal.board.run)))
            save_checkpoint(dqn_snake_normal, model_path)

        if (i % 250 == 0):
            test_snake(
                board=board_food,
                snake=dqn_snake_TEST,
                observation_function = lambda: observation_cat(
                    observation_near(
                        board=board_food,
                        snake=dqn_snake_TEST,
                        kernel=array([state_size, state_size])
                    ),
                    observation_to_bool(observation_food(dqn_snake_TEST))
                )
            )

    # test a snake over 10 displayed runs
    # for _ in range(10):
    #     display_run(board_food_mine, dqn_snake_normal, array([state_size, state_size]), pretty_print, lambda: observation_near(board=board, snake=snake, kernel=array([state_size, state_size])))