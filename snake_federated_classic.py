# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import pretty_print, display, display_run
from composed_functions.dqn_helper import dqn, load_checkpoint, load_checkpoint_to_snake, save_checkpoint
from federated_learning.federated_average import agregate

if __name__ == "__main__":
    '''
    Trains multiple seperated DQN-agents in seperate environments with different rules
    Does a fedaverage between the agents to show that multiple rules for multiple agents can increase preformance/exploration
    '''

    episode_amount: int = 500
    env_episode_amount: int = 100
    save_every: int = 5
    board_dim: int = 25
    state_size: int = 7
    model_id: str = "{}x{}".format(state_size, state_size)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    # snake for:
    #           * early game exploration (small snake length and high epsilon)
    #           * late game exploration (large snake and high epsilon)
    dqn_snake_exploration: DQNAgent = DQNAgent(
        state_size          = state_size**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 20]),
        seed                = 69,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0.25, 100_000),
        learning_rate       = 1e-4,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 1_000_000
    )

    #           * early game exploitation (small snake length and low epsilon)
    #           * late game exploitation (large snake and low epsilon)
    dqn_snake_exploitation: DQNAgent = DQNAgent(
        state_size          = state_size**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 20]),
        seed                = 69,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(0.25, 0., 50_000),
        learning_rate       = 1e-4,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 1_000_000
    )

    #           * normal snake agent (small snake with decreesing epsilon)
    dqn_snake_normal: DQNAgent = DQNAgent(
        state_size          = state_size**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 69,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0., 25_000),
        learning_rate       = 1e-4,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 1_000_000
    )

    #           * board with static size and lots of fruit
    board_LOTS_of_fruit: Board = Board(
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
    #           * default board with fixed size and only one fruit
    board_fruit_mine: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(spawn_amount = array([1, 1]))
        },
    )

    # Lists holding all snakes and boards for easy itteration and mixing
    snakes: list = [dqn_snake_exploration, dqn_snake_exploitation, dqn_snake_normal]
    boards: list = [board_LOTS_of_fruit, board_fruit_mine]

    checkpoint = load_checkpoint(model_path)
    load_checkpoint_to_snake(dqn_snake_normal, checkpoint)
    load_checkpoint_to_snake(dqn_snake_exploitation, checkpoint)
    load_checkpoint_to_snake(dqn_snake_exploration, checkpoint)

    for i in range(episode_amount):
        # train all snakes on all boards for env_episode_amount episodes
        median: float = 0 # median length of dqn_snake_normal in board_fruit_mine
        for snake in snakes:
            for board in boards:
                board.set_snakes([snake])
                median = dqn(board, snake, env_episode_amount, lambda: observation_near(board=board, snake=snake, kernel=array([state_size, state_size])))
                
        print('\rEpisode {}\tAverage Scores {:.3f}\tRandom act chance {:.6f}'.format(i * env_episode_amount, median, dqn_snake_normal.epsilon(dqn_snake_normal.board.run)))

        # do a fedaverage between them
        agregate(snakes)

        # Save their model
        if i % save_every == 0:
            save_checkpoint(dqn_snake_normal, model_path)

    # test a snake over 10 displayed runs
    for _ in range(10):
        display_run(board_fruit_mine, dqn_snake_normal, array([state_size, state_size]), pretty_print, observation_near)