# Math modules
from numpy import array, arange
from random import random, choice, randrange

# Repo imports
from snake_env.snake_agents.virtual_snake import *
from snake_env.snake_agents.observation_functions import *
from snake_env.snake_agents.neural_nets.deep_q_learning import *

# Torch imports
from torch import no_grad, argmax, LongTensor
from torch.optim import Adam

# Operating system
from os import name as system_name

class Agent(Snake):
    '''
    Placeholders that all agents should inherrent from.
    '''
    def __init__(self, init_snake_lengths: array) -> None:
        '''
        Initialize a Agent object.
        '''
        super(Agent, self).__init__(init_snake_lengths)
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
    '''
    def __init__(self, init_snake_lengths: array = array([2, 2])) -> None:
        '''
        Initialize a RandomAgent object.
        '''
        super(RandomAgent, self).__init__(init_snake_lengths)

    def act(self) -> int:
        '''
        Returns random action
        '''
        return randrange(len(self.all_actions)), True

class ControllableAgent(Agent):
    '''
    Waits for player input to act in the environemnt
    '''
    def __init__(self, init_snake_lengths: array = array([2, 2])) -> None:
        '''
        Initialize a ControllableAgent object.
        '''
        super(ControllableAgent, self).__init__(init_snake_lengths)

    def act(self) -> int:
        '''
        Returns action given key presses
        '''
        def KeyCheck() -> int:
            ''' Input only windows supported '''
            assert system_name == 'nt', "Input is only windows supported, sorry Linux King"
            from msvcrt import getch

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

        return KeyCheck(), False

class DQNAgent(Agent):    
    '''
    Implements DQN (deep q learning) to act in given enviroment
    '''
    def __init__(self, state_size: int, action_size: int, init_snake_lengths: array = array([2, 2]), seed: int = 1337, batch_size: int = 64, gamma: float = 0.999, epsilon: Epsilon = Epsilon(1, 0, 50_000), learning_rate: float = 5e-4, tau: float = 5e-4, update_every: int = 10, buffer_size: int = 500_000) -> None:
        '''
        Initialize an Agent object.
        '''
        super(DQNAgent, self).__init__(init_snake_lengths)

        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.epsilon: Epsilon = epsilon
        self.learning_rate: float = learning_rate
        self.tau: float = tau
        self.update_every: int = update_every
        self.buffer_size: int = buffer_size

        self.state_size = state_size
        self.action_size = action_size
        
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

    def step(self, state: FloatTensor, action: LongTensor, reward: FloatTensor, next_state: FloatTensor, done: bool) -> None:
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
    
    def act(self, state: FloatTensor) -> int:
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
        eps_threshold: float = self.epsilon(self.board.run)

        if sample > eps_threshold:
            return argmax(action_values.cpu().data), False
        else:
            return LongTensor([choice(arange(self.action_size))])[0], True
            
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
