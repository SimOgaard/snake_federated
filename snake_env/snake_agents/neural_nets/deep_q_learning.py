# Math modules
from random import sample

# Generic modules
from collections import namedtuple, deque

# Torch imports
from torch import FloatTensor
from torch import nn
from torch.nn import functional as F
from torch import device as torch_device
from torch.cuda import is_available as cuda_is_available

# Representing a single transition in our environment (mapped state, action to next_state, reward)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Global device 
device: str = torch_device("cuda" if cuda_is_available() else "cpu")

class ReplayMemory(object):
    '''
    Cyclic buffer of bounded size holding multiple transitions
    '''
    def __init__(self, capacity) -> None:
        self.memory = deque([],maxlen=capacity)

    def push(self, *args) -> None:
        '''
        Saves a transition
        '''
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> list:
        '''
        Selects a random batch of transitions for training
        '''
        return sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class DQNConv(nn.Module):
    '''
    DQN that implemets a convolutional neural network
    '''
    def __init__(self, h: int, w: int, output_size: int) -> None:
        super(DQNConv, self).__init__()
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

    # Called with either one element to determine next action, or a batch during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x) -> FloatTensor:
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQNLin(nn.Module):
    '''
    DQN that implemets a linear neural network
    '''
    def __init__(self, input_size: int, output_size: int) -> None:
        super(DQNLin, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 128)
        self.linear6 = nn.Linear(128, 64)
        self.linear7 = nn.Linear(64, 32)
        self.head    = nn.Linear(32, output_size)

    # Called with either one element to determine next action, or a batch during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x) -> FloatTensor:
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = F.relu(self.linear7(x))
        return F.relu(self.head(x))