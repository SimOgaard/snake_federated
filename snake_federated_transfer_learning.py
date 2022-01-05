# Repo imports
import torch
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *
from snake_terminal import pretty_print, display
from snake_testing import display_run
from snake_training import load_checkpoint

# Math modules
from numpy import mean as numpy_mean

# Torch imports
from torch import save, load
from torch import div as torch_div

# Generic imports
from os import path

def dqn(board: Board, snake: Snake, env_episode_amount: int, observation_function: object) -> float:
    scores_window = deque(maxlen=100) # last 100 scores

    for i in range (env_episode_amount):
        board.__restart__() # restart board
        state: FloatTensor = observation_function() # save init state
        while board.is_alive(): # check if snakes are alive

            action, is_random = snake.act(state) # choose an action for given snake
            reward: float = snake.move(action, is_random)

            action = LongTensor([action]) # take the agents action that leed to that reward and state
            reward = FloatTensor([reward]) # take the reward that the agent stored

            next_state: FloatTensor = observation_function() # observe what steps taken lead to

            snake.step(state, action, reward, next_state, snake.done) # signal step to snake

            state = next_state # set old state to the next state

        scores_window.append(len(snake.snake_body)) # save the most recent score
        #print('\rEpisode {}\tAverage Score {:.3f}\tRandom act chance {:.6f}'.format(board.run, numpy_mean(scores_window), snake.epsilon(snake.board.run)), end="")

    return numpy_mean(scores_window)
    #print('\rEpisode {}\tAverage Score {:.3f}\tRandom act chance {:.6f}'.format(board.run, numpy_mean(scores_window), snake.epsilon(snake.board.run)))

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

    episode_amount: int = 10_000
    env_episode_amount: int = 100
    save_every: int = 50
    board_dim: int = 20
    state_size: int = 5
    model_id: str = "{}x{}".format(state_size, state_size)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    # Snake and its environment with only mines
    dqn_snake_mine: DQNAgent = DQNAgent(
        state_size          = state_size**2, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0., 100_000),
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
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "mine_tile": MineTile()
        },
    )

    # Snake and its environment with fruit
    dqn_snake_fruit: DQNAgent = DQNAgent(
        state_size          = state_size**2, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 69,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0., 100_000),
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
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile()
        },
    )

    if (path.exists(model_path)): # load model
        checkpoint = load(model_path)
        load_checkpoint(dqn_snake_mine, checkpoint)
        load_checkpoint(dqn_snake_fruit, checkpoint)

    for i in range(episode_amount):
        # train each snake seperatly for env_episode_amount episodes
        mine_median: float = dqn(board_mine, dqn_snake_mine, env_episode_amount, lambda: observation_near(board=board_mine, snake=dqn_snake_mine, kernel=array([state_size, state_size])))
        fruit_median: float = dqn(board_fruit, dqn_snake_fruit, env_episode_amount, lambda: observation_near(board=board_mine, snake=dqn_snake_mine, kernel=array([state_size, state_size])))

        print('\rEpisode {}\tAverage Scores ({:.3f}, {:.3f})\tRandom act chance {:.6f}'.format(i * env_episode_amount, mine_median, fruit_median, dqn_snake_fruit.epsilon(dqn_snake_fruit.board.run)))

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
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(),
            "mine_tile": MineTile()
        },
    )
    # display 10 runs
    for _ in range(10):
        display_run(board_combined, dqn_snake_mine, array([board_dim+2, board_dim+2]))
