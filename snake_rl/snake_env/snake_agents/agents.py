# Math modules
from numpy import array, arange
from random import seed as random_seed
from random import random, choice
from math import exp

# Input
from msvcrt import getch

# Repo imports
from snake_env.snake_agents.virtual_snake import *
from snake_env.snake_agents.observation_functions import *
from snake_env.snake_agents.neural_nets.deep_q_learning import *

# Torch imports
from torch import no_grad, argmax
from torch.optim import Adam

class Agent(Snake):
    '''
    Placeholders that all agents should inherrent from.
    '''
    def __init__(self) -> None:
        '''
        Initialize a Agent object.
        '''
        super(Agent, self).__init__()
        self.action = 0
        self.observation = None
        self.reward = None

    def __restart__(self) -> None:
        '''
        Restarts self.
        '''
        super(Agent, self).__restart__()

class RandomAgent(Agent):
    '''
    Random action from all possible actions our snake can perform
    Default observation function is ObservationNone
    Ovveride it by doing RandomAgent.observate = ObservationRaycast.observate to test different observation functions
    '''
    def act(self) -> int:
        '''
        Returns random action
        '''
        return randrange(len(self.all_actions))

class ControllableAgent(Agent):
    '''
    Waits for player input to act in the environemnt
    '''
    def act(self) -> int:
        '''
        Returns action given key presses
        '''
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

        return KeyCheck()

class DQNAgent(Agent):    
    '''
    Implements DQN (deep q learning) to act in given enviroment
    '''
    def __init__(self, state_size: int, action_size: int, seed: int = 1337, batch_size: int = 64, gamma: float = 0.999, epsilon_start: float = 1, epsilon_end: float = 0.0, epsilon_decay: int = 2500, learning_rate: float = 5e-4, tau: float = 1e-3, update_every: int = 10, buffer_size: int = 500_000) -> None:
        '''
        Initialize an Agent object.
        '''
        super(DQNAgent, self).__init__()

        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.epsilon_start: float = epsilon_start
        self.epsilon_end: float = epsilon_end
        self.epsilon_decay: int = epsilon_decay
        self.learning_rate: float = learning_rate
        self.tau: float = tau
        self.update_every: int = update_every
        self.buffer_size: int = buffer_size

        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed(seed)
        
        #Q Network
        self.qnetwork_local = DQNLin(state_size, action_size, seed).to(device)
        self.qnetwork_target = DQNLin(state_size, action_size, seed).to(device)
        
        self.optimizer = Adam(self.qnetwork_local.parameters(),lr=learning_rate)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def __restart__(self) -> None:
        '''
        Restarts self
        '''
        super(DQNAgent, self).__restart__()

    def step(self, state: FloatTensor, action: FloatTensor, reward: FloatTensor, next_state: FloatTensor, done: bool) -> None:
        '''
        Trains a step
        '''
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>self.batch_size:
                experience = self.memory.sample()
                self.learn(experience)
    
    def calculate_epsilon(self) -> float:
        '''
        Returns current epsilon value 
        '''
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * exp(-1. * self.board.run / self.epsilon_decay)

    def act(self, state: FloatTensor) -> None:
        '''
        Returns action for given state as per current policy
        '''
        new_state: FloatTensor = state.unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with no_grad():
            action_values = self.qnetwork_local(new_state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        sample: float = random()
        eps_threshold: float = self.calculate_epsilon()

        if sample > eps_threshold:
            return argmax(action_values.cpu().data)
        else:
            return choice(arange(self.action_size))
            
    def learn(self, experiences) -> None:
        """
        Update value parameters using given batch of experience tuples.
        """
        states, actions, rewards, next_state, dones = experiences
        criterion = nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (self.gamma * labels_next*(1-dones))
        
        loss = criterion(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local,self.qnetwork_target)
            
    def soft_update(self, local_model, target_model) -> None:
        '''
        Soft update model parameters.
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)