# Math modules
from random import sample
from random import seed as random_seed

# Generic modules
from collections import namedtuple, deque

# Torch imports
from torch import FloatTensor, manual_seed, nn, vstack
from torch.nn import functional as F
from torch import device as torch_device
from torch.cuda import is_available as cuda_is_available

# Global stored string of current utilized device 
device: str = torch_device("cuda" if cuda_is_available() else "cpu")

class ReplayBuffer:
    '''
    Fixed -size buffe to store experience tuples.
    '''
    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int = 1337) -> None:
        '''
        Initialize a ReplayBuffer object.
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random_seed(seed)
        
    def add(self, state: FloatTensor, action: FloatTensor, reward: FloatTensor, next_state: FloatTensor, done: bool) -> None:
        '''
        Add a new experience to memory
        '''
        e = self.experiences(state,action,reward,next_state,FloatTensor([done]))
        self.memory.append(e)
        
    def sample(self) -> tuple:
        '''
        Randomly sample a batch of experiences from memory
        '''
        experiences = sample(self.memory,k=self.batch_size)
        
        states = vstack([e.state for e in experiences if e is not None]).to(device)
        actions = vstack([e.action for e in experiences if e is not None]).to(device)
        rewards = vstack([e.reward for e in experiences if e is not None]).to(device)
        next_states = vstack([e.next_state for e in experiences if e is not None]).to(device)
        dones = vstack([e.done for e in experiences if e is not None]).to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        '''
        Return the current size of internal memory
        '''
        return len(self.memory)

class DQNCnn(nn.Module):
    '''
    DQN that implemets a convolutional neural network
    '''
    def __init__(self, h: int, w: int, output_size: int, seed: int = 1337) -> None:
        '''
        Initialize a cnn object.
        '''
        super(DQNCnn, self).__init__()
        self.seed = manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, output_size)

    def forward(self, x: FloatTensor) -> FloatTensor:
        '''
        Called with either one element to determine next action, or a batch during optimization. Returns tensor([[left0exp,right0exp]...]).
        '''
        # x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQNLin(nn.Module):
    '''
    DQN that implemets a linear Actor (Policy) Model.
    '''
    def __init__(self, state_size: int, action_size: int, seed: int = 1337) -> None:
        '''
        Initialize a linear neural net.
        '''
        super(DQNLin, self).__init__()
        self.seed = manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, action_size)
        
    def forward(self, x) -> FloatTensor:
        '''
        Build a network that maps state -> action values.
        '''
        # x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)