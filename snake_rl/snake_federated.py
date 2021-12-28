# Repo imports
import torch
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *
from snake_terminal import pretty_print, display
from snake_testing import display_run

# Math modules
from numpy import mean as numpy_mean

# Torch imports
from torch import save, load
from torch import div as torch_div

# Generic imports
from os import path
from copy import deepcopy

def dqn(board: Board, snake: Snake):
    scores_window = deque(maxlen=100) # last 100 scores

    for i in range (env_episode_amount):
        board.__restart__() # restart board
        state: FloatTensor = observation_full(board = board) # save init state
        while board.is_alive(): # check if snakes are alive

            action: int = snake.act(state) # choose an action for given snake
            reward: float = snake.move(action)

            action = LongTensor([action]) # take the agents action that leed to that reward and state
            reward = FloatTensor([reward]) # take the reward that the agent stored

            next_state: FloatTensor = observation_full(board = board) # observe what steps taken lead to

            snake.step(state, action, reward, next_state, snake.done) # signal step to snake

            state = next_state # set old state to the next state

        scores_window.append(len(snake.snake_body)) # save the most recent score
        print('\rEpisode {}\tAverage Score {:.3f}\tRandom act chance {:.6f}'.format(board.run, numpy_mean(scores_window), snake.calculate_epsilon()), end="")
        
    print('\rEpisode {}\tAverage Score {:.3f}\tRandom act chance {:.6f}'.format(board.run, numpy_mean(scores_window), snake.calculate_epsilon()))

def agregate(agents: list) -> None:
    '''
    Agregates between each agent in agents and each model in agent
    '''
    def agregate_model(models: list):
        '''
        Returns agregated median of all models
        '''
        with torch.no_grad():
            model_average = dict(models[0].named_parameters())

            for key in model_average.keys():
                # print(model_average[key].data)
                for i in range(1, len(models)):
                    # print(dict(models[i].named_parameters())[key].data)
                    model_average[key].data += dict(models[i].named_parameters())[key].data#.clone()

                model_average[key].data = torch_div(model_average[key].data, len(models))
                # print(model_average[key].data)

        return model_average

    # agregate both models
    agregated_qnetwork_local = agregate_model([agent.qnetwork_local for agent in agents])
    agregated_qnetwork_target = agregate_model([agent.qnetwork_target for agent in agents])

    # apply a copied version of the returning agregated model to each agent
    for agent in agents:
        agent.qnetwork_local.load_state_dict(agregated_qnetwork_local)
        agent.qnetwork_target.load_state_dict(agregated_qnetwork_target)

    # ### TESTING
    # print((agents[0].qnetwork_local.fc1.weight == agents[1].qnetwork_local.fc1.weight).all()) # SHOULD PRINT TRUE
    # # Manipulate param
    # with torch.no_grad():
    #     agents[0].qnetwork_local.fc1.weight.zero_()

    # print((agents[0].qnetwork_local.fc1.weight == agents[1].qnetwork_local.fc1.weight).all()) # SHOULD PRINT FALSE
    # ### TESTING

if __name__ == "__main__":
    '''
    Trains two seperated DQN-agents in two seperate environments with different rules
    Does a fedaverage between the two agents to show that two agents in seperate environments can transfer what they have learned to eachother
    '''

    episode_amount: int = 1_000
    env_episode_amount: int = 100
    save_every: int = 5_000
    board_dim: int = 5
    model_id: str = "{}x{}".format(board_dim + 2, board_dim + 2)
    model_path: str = 'snake_rl/models/fed_checkpoint{}.pth'.format(model_id)

    # Snake and its environment with only mines
    dqn_snake_mine: DQNAgent = DQNAgent(
        state_size          = (board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
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
    board_mine: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_mine],
        tiles_populated         = [MineTile],
    )

    # Snake and its environment with fruit
    dqn_snake_fruit: DQNAgent = DQNAgent(
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
    board_fruit: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_fruit],
        tiles_populated         = [FoodTile],
    )

    if (path.exists(model_path)): # load model
        checkpoint = load(model_path)
        dqn_snake_mine.qnetwork_local.load_state_dict(checkpoint['network_local'])
        dqn_snake_mine.qnetwork_target.load_state_dict(checkpoint['network_target'])

        dqn_snake_fruit.qnetwork_local.load_state_dict(checkpoint['network_local'])
        dqn_snake_fruit.qnetwork_target.load_state_dict(checkpoint['network_target'])

    for i in range(episode_amount):
        # train each snake seperatly for env_episode_amount episodes
        dqn(board=board_mine, snake=dqn_snake_mine)
        dqn(board=board_fruit, snake=dqn_snake_fruit)

        # do a fedaverage between them
        agregate([dqn_snake_mine, dqn_snake_fruit])

        # Save their model
        if i % save_every == 0:
            state: dict = {
                'network_local': dqn_snake_mine.qnetwork_local.state_dict(),
                'network_target': dqn_snake_mine.qnetwork_target.state_dict()
            }
            save(state, model_path)

    # test the snake that hadnt seen any fruits on fruit board and see how it reacts
    board_fruit.snakes = [dqn_snake_mine]
    display_run(board_fruit, dqn_snake_mine)
