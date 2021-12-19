# Math modules
from numpy import array
from random import random
from math import exp

# Input
from msvcrt import getch

# Repo imports
from snake_env.snake_agents.virtual_snake import *
from snake_env.snake_agents.observation_functions import *
from snake_env.snake_agents.neural_nets.deep_q_learning import *

# Torch imports
from torch import tensor, no_grad, cat, argmax
from torch import zeros as torch_zeros

from torch import long as torch_tensor_long
from torch import bool as torch_tensor_bool

from torch.optim import RMSprop


# EVERY SNAKE TYPE IE:
# player controlled
# randomized
# nn

class Agent(Snake):
    def __init__(self) -> None:
        super(Agent, self).__init__()
        self.action = 0
        self.observation = None
        self.reward = None

    def __restart__(self) -> None:
        super(Agent, self).__restart__()

    def get_step(self) -> tuple:
        return self.action, self.observation, self.reward, self.done

class RandomAgent(Agent, ObservationFull):
    '''
    Random action from all possible actions our snake can perform
    Default observation function is ObservationNone
    Ovveride it by doing RandomAgent.observate = ObservationRaycast.observate to test different observation functions
    '''
    def __iter__(self) -> None:
        self.__restart__()
        return self

    def __next__(self) -> None:
        # is the same as:
        # snake.move(snake.next_action())
        self.action: int = randrange(len(self.all_actions))
        self.reward: float = self.move(self.action)

class ControllableAgent(Agent, ObservationFull):
    '''
    Waits for player input to act in the environemnt
    '''
    def __iter__(self) -> None:
        self.__restart__()
        return self

    def __next__(self) -> None:
        def KeyCheck() -> array:
            global Break_KeyCheck
            Break_KeyCheck = False
            
            base = getch()
            if base == b'\x00':
                sub = getch()
                if sub == b'H':
                    return 1
                elif sub == b'M':
                    return 2
                elif sub == b'P':
                    return 0
                elif sub == b'K':
                    return 3

        print(self.board.board)

        self.action: int = KeyCheck()
        self.reward: float = self.move(self.action)

class DQNAgent(Agent, ObservationNear):
    '''
    Implements DQN (deep q learning) to act in given enviroment
    '''
    def __init__(self, policy_net: DQNLin, target_net: DQNLin, batch_size: int = 1, gamma: float = 0.999, epsilon_start: float = 0.9, epsilon_end: float = 0.001, epsilon_decay: int = 1000, target_update: int = 10) -> None:
        super(DQNAgent, self).__init__()

        self.policy_net = policy_net
        self.target_net = target_net

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = RMSprop(policy_net.parameters())
        self.memory = ReplayMemory(1000)

        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.epsilon_start: float = epsilon_start
        self.epsilon_end: float = epsilon_end
        self.epsilon_decay: int = epsilon_decay
        self.target_update: int = target_update

    def __restart__(self) -> None:
        super(DQNAgent, self).__restart__()

    def __iter__(self) -> None:
        self.__restart__()
        return self

    def __next__(self) -> None:
        sample: float = random()
        eps_threshold: float = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            exp(-1. * self.board.run / self.epsilon_decay)

        if sample > eps_threshold:
            with no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.action: int = argmax(self.policy_net(self.observation))
        else:
            self.action: int = tensor([[randrange(len(self.all_actions))]], device=device, dtype=torch_tensor_long)

        self.reward: float = self.move(self.action)

    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        #print("lamao")

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch_tensor_bool)
        non_final_next_states = cat([s for s in batch.next_state if s is not None])
        state_batch = cat(batch.state)
        action_batch = cat(batch.action)
        reward_batch = cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(0, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch_zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = argmax(self.target_net(non_final_next_states)).detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()