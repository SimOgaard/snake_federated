# Torch imports
from torch import save, load

# Os system modules
from os import path

# Repo imports
from snake_env.snake_agents.neural_nets.deep_q_learning import *

def load_checkpoint(model_path) -> dict:
    '''
    Load model checkpoint given path 
    '''
    if (path.exists(model_path)):
        return load(model_path, map_location=device)
    return None

def load_checkpoint_to_snake(snake, checkpoint) -> None:
    '''
    Loads given checkpoint to snake
    '''
    if (checkpoint != None):
        snake.qnetwork_local.load_state_dict(checkpoint['network_local'])
        snake.qnetwork_target.load_state_dict(checkpoint['network_target'])

def save_checkpoint(snake, model_path: str) -> None:
    '''
    Saves dqn snake networks
    '''
    state: dict = {
        'network_local': snake.qnetwork_local.state_dict(),
        'network_target': snake.qnetwork_target.state_dict()
    }
    save(state, model_path)

def clone_snake(_from, _to):
    '''
    Clones snake networks from snake _from to snake _to
    '''
    state: dict = {
        'network_local': _from.qnetwork_local.state_dict(),
        'network_target': _from.qnetwork_target.state_dict()
    }
    _to.qnetwork_local.load_state_dict(state['network_local'])
    _to.qnetwork_target.load_state_dict(state['network_target'])

# Torch imports
from torch import FloatTensor, LongTensor
# Math module
from numpy import mean as numpy_mean
# Generic modules
from collections import deque

def dqn(board, snake, env_episode_amount: int, observation_function: object) -> float:
    '''
    Trains snake on board for env_episode_amount episodes
    '''
    scores_window = deque(maxlen=100) # last 100 scores

    board.set_snakes(snake)

    for _ in range (env_episode_amount):
        board.__restart__() # restart board
        state: FloatTensor = observation_function() # save init state
        while board.is_alive(): # check if snakes are alive

            action, is_random = snake.act(state) # choose an action for given snake
            reward: float = snake.move(action, is_random)

            next_state: FloatTensor = observation_function() # observe what steps taken lead to

            snake.step(state, action, reward, next_state, snake.done) # signal step to snake

            state = next_state # set old state to the next state

        scores_window.append(len(snake.snake_body)) # save the most recent score

    return numpy_mean(scores_window)