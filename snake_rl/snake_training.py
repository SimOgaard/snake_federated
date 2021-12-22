# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

# Math modules
from numpy import mean as numpy_mean

# Torch imports
from torch import tensor, save, load

# Generic imports
from os import path

if __name__ == "__main__":
    '''
    Trains a DQN-agent
    '''

    episode_amount: int = 100_000
    save_every: int = 5_000
    board_dim: int = 5
    model_id: str = "{}x{}".format(board_dim + 2, board_dim + 2)
    model_path: str = 'snake_rl/models/checkpoint{}.pth'.format(model_id)

    dqn_snake: DQNAgent = DQNAgent(
        state_size          = (board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 10]),
        seed                = 1337,
        batch_size          = 64,
        gamma               = 0.999,
        epsilon_start       = 1.,
        epsilon_end         = 0.,
        epsilon_decay       = 10_000,
        learning_rate       = 5e-4,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 1_000_000
    )

    board: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake],
    )

    scores_window = deque(maxlen=100) # last 100 scores

    if (path.exists(model_path)): # load model
        checkpoint = load(model_path)
        dqn_snake.qnetwork_local.load_state_dict(checkpoint['network_local'])
        dqn_snake.qnetwork_target.load_state_dict(checkpoint['network_target'])

    while board.run < episode_amount:
        
        board.__restart__() # restart board
        state: FloatTensor = observation_full(board = board) # save init state

        while board.is_alive(): # check if snakes are alive

            action: int = dqn_snake.act(state) # choose an action for given snake
            reward: float = dqn_snake.move(action)

            action = tensor([action], device=device) # take the agents action that leed to that reward and state
            reward = tensor([reward], device=device) # take the reward that the agent stored

            next_state: FloatTensor = observation_full(board = board) # observe what steps taken lead to

            dqn_snake.step(state, action, reward, next_state, dqn_snake.done) # signal step to snake

            state = next_state # set old state to the next state

        scores_window.append(len(dqn_snake.snake_body)) # save the most recent score
        print('\rEpisode {}\tAverage Score {:.3f}\tRandom act chance {:.6f}'.format(board.run, numpy_mean(scores_window), dqn_snake.calculate_epsilon()), end="")
        
        if board.run != 0:
            if board.run % 100 == 0:
                print('\rEpisode {}\tAverage Score {:.3f}\tRandom act chance {:.6f}'.format(board.run, numpy_mean(scores_window), dqn_snake.calculate_epsilon()))
            if board.run % save_every == 0:
                state: dict = {
                    'network_local': dqn_snake.qnetwork_local.state_dict(),
                    'network_target': dqn_snake.qnetwork_target.state_dict()
                }
                save(state, model_path)