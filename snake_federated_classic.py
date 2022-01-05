# Repo imports
import torch
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *
from snake_terminal import pretty_print, display
from snake_testing import display_run
from snake_training import load_checkpoint
from snake_federated_transfer_learning import dqn, agregate

# Math modules
from numpy import mean as numpy_mean

# Torch imports
from torch import save, load
from torch import div as torch_div

# Generic imports
from os import path

if __name__ == "__main__":
    '''
    Trains multiple seperated DQN-agents in seperate environments with different rules
    Does a fedaverage between the agents to show that multiple rules for multiple agents can increase preformance/exploration
    '''

    episode_amount: int = 500
    env_episode_amount: int = 100
    save_every: int = 50
    board_dim: int = 5
    state_size: int = 5
    model_id: str = "{}x{}".format(state_size, state_size)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    # snake for:
    #           * early game exploration (small snake length and high epsilon)
    #           * late game exploration (large snake and high epsilon)
    dqn_snake_exploration: DQNAgent = DQNAgent(
        state_size          = state_size**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 20]),
        seed                = 1337,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon_start       = 1.,
        epsilon_end         = 0.25,
        epsilon_decay       = 100_000,
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
        epsilon_start       = .25,
        epsilon_end         = 0.,
        epsilon_decay       = 100_000,
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
        epsilon_start       = 1.,
        epsilon_end         = 0.,
        epsilon_decay       = 10_000,
        learning_rate       = 1e-4,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 1_000_000
    )

    # different board types:
    #           * board with dynamic size and only food tile
    board_fruit: Board = Board(
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
    #           * board with dynamic size and only mine tile
    board_mine: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "mine_tile": MineTile()
        },
    )
    #           * default board with fixed size both mine and food tile
    board_fruit_mine: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(),
            "mine_tile": MineTile()
        },
    )

    # Lists holding all snakes and boards for easy itteration and mixing
    snakes: list = [dqn_snake_exploration, dqn_snake_exploitation, dqn_snake_normal]
    boards: list = [board_fruit, board_mine, board_fruit_mine]

    if (path.exists(model_path)): # load model
        checkpoint = load(model_path)
        load_checkpoint(dqn_snake_exploration, checkpoint)
        load_checkpoint(dqn_snake_exploitation, checkpoint)
        load_checkpoint(dqn_snake_normal, checkpoint)

    for i in range(episode_amount):
        # train all snakes on all boards for env_episode_amount episodes
        median: float = 0 # median length of dqn_snake_normal in board_fruit_mine
        for snake in snakes:
            for board in boards:
                board.set_snakes([snake])
                median = dqn(board, snake, env_episode_amount, lambda: observation_near(board=board, snake=snake, kernel=array([5, 5])))
                
        print('\rEpisode {}\tAverage Scores {:.3f}\tRandom act chance {:.6f}'.format(i * env_episode_amount, median, dqn_snake_normal.calculate_epsilon()))

        # do a fedaverage between them
        agregate(snakes)

        # Save their model
        if i % save_every == 0:
            state: dict = {
                'network_local': dqn_snake_normal.qnetwork_local.state_dict(),
                'network_target': dqn_snake_normal.qnetwork_target.state_dict()
            }
            save(state, model_path)

    # test a snake over 10 displayed runs
    for _ in range(10):
        display_run(board_fruit_mine, dqn_snake_normal, array([board_dim+2, board_dim+2]))
