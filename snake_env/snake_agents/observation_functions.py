# Pytorch tensors
from torch import FloatTensor, flatten
import torch

class ObservationNone():
    '''
    Default observation state: None
    '''
    def observate(self) -> None:
        self.observation = None
        return self.observation

class ObservationNear():
    '''
    Returns dumbed down state around head and food position
    '''
    def observate(self) -> tuple:
        head = self.snake_body[0]

        point_u = head + self.all_actions[1]
        point_r = head + self.all_actions[2]
        point_d = head + self.all_actions[0]
        point_l = head + self.all_actions[3]

        try:

            state = [
                # What is up
                self.board.board[point_u[0]][point_u[1]] == 0. or self.board.board[point_u[0]][point_u[1]] == 3.,
                
                # What is right
                self.board.board[point_r[0]][point_r[1]] == 0. or self.board.board[point_r[0]][point_r[1]] == 3.,

                # What is down
                self.board.board[point_d[0]][point_d[1]] == 0. or self.board.board[point_d[0]][point_d[1]] == 3.,

                # What is left
                self.board.board[point_l[0]][point_l[1]] == 0. or self.board.board[point_l[0]][point_l[1]] == 3.,
                
                # Food location 
                # game.food.x < game.head.x,  # food up
                # game.food.x > game.head.x,  # food right
                # game.food.y < game.head.y,  # food down
                # game.food.y > game.head.y,  # food left

                # Current_direction
                self.action
            ]

            self.observation = FloatTensor(state)
            return self.observation
        except:
            self.observation = FloatTensor([5,5,5,5,self.action])
            return self.observation

class ObservationFull():
    '''
    Returns the whole board
    '''
    def observate(self) -> FloatTensor:
        self.observation = flatten(self.board.board.detach().clone())
        return self.observation
