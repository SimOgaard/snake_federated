# Repo imports
import torch
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *
from snake_terminal import pretty_print, display
from snake_testing import display_run
from snake_training import load_checkpoint
from snake_federated import dqn, agregate

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
    Does a fedaverage between the agents to show that multiple rules for multiple agents can increase preformance
    '''

    episode_amount: int = 500
    env_episode_amount: int = 100
    save_every: int = 50
    board_dim: int = 5
    model_id: str = "{}x{}".format(board_dim + 2, board_dim + 2)
    model_path: str = 'models/fed_checkpoint{}.pth'.format(model_id)

    # snake for:
    #           * early game exploration (small snake length and high epsilon)
    #           * late game exploration (large snake and high epsilon)
    dqn_snake_exploration: DQNAgent = DQNAgent(
        state_size          = (board_dim + 2)**2,
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
        state_size          = (board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
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

    #           * normal snake agent
    dqn_snake_normal: DQNAgent = DQNAgent(
        state_size          = (board_dim + 2)**2,
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

    # different board types
    board_fruit: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_exploration, dqn_snake_exploitation, dqn_snake_normal],
        tiles_populated         = [FoodTile],
    )
    board_mine: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_exploration, dqn_snake_exploitation, dqn_snake_normal],
        tiles_populated         = [MineTile],
    )
    board_fruit_mine: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_exploration, dqn_snake_exploitation, dqn_snake_normal],
        tiles_populated         = [FoodTile, MineTile],
    )

    if (path.exists(model_path)): # load model
        checkpoint = load(model_path)
        load_checkpoint(dqn_snake_exploration, checkpoint)
        load_checkpoint(dqn_snake_exploitation, checkpoint)
        load_checkpoint(dqn_snake_normal, checkpoint)

    for i in range(episode_amount):
        # train each snake seperatly for env_episode_amount episodes
        mine_median: float = dqn(board=board_mine, snake=dqn_snake_mine)
        fruit_median: float = dqn(board=board_fruit, snake=dqn_snake_fruit)

        print('\rEpisode {}\tAverage Scores ({:.3f}, {:.3f})\tRandom act chance {:.6f}'.format(i * env_episode_amount, mine_median, fruit_median, dqn_snake_fruit.calculate_epsilon()))

        # do a fedaverage between them
        agregate([dqn_snake_mine, dqn_snake_fruit])

        # Save their model
        if i % save_every == 0:
            state: dict = {
                'network_local': dqn_snake_mine.qnetwork_local.state_dict(),
                'network_target': dqn_snake_mine.qnetwork_target.state_dict()
            }
            save(state, model_path)

    # test the snake that hadnt seen any fruits only mines on fruit and mine board and see how it reacts
    board_combined: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_mine],
        tiles_populated         = [FoodTile, MineTile],
    )
    # display 10 runs
    for _ in range(10):
        display_run(board_combined, dqn_snake_mine)
