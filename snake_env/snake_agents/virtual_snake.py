# Math modules
from random import shuffle, randint

# Torch modules
from torch import FloatTensor

# Repo imports
from snake_env.tiles.tiles import *

class Snake():
    '''
    Snake (self explanatory i guess)
    '''

    def __init__(self, init_snake_lengths: array, snake_tiles: dict = {"snake_tile": SnakeTile(), "snake_head_tile": SnakeHeadTile()}) -> None:
        '''
        Initilizes our snake at given board
        '''
        super(Snake, self).__init__()

        self.snake_move_count: int = 1
        self.all_actions: list = [array([1, 0]), array([-1, 0]), array([0, 1]), array([0, -1])]
        self.board = None
        self.init_snake_lengths: array = init_snake_lengths - array([1, 1])
        self.snake_tiles = snake_tiles

        self.death = 0
        self.random_action_death = 0

        # needs to run restart but you do it in externally
        # self.__restart__()

    def __restart__(self) -> None:
        '''
        Places our snake on board and restarts every value
        '''

        def create_snake() -> tuple:
            '''
            Chooses random point inside board_shape and places snake at that point in random directions of random length given self.init_snake_lengths
            Returns snake body list and direction
            '''
            # choose random max length that snake should be init as
            rand_max_length: int = randint(self.init_snake_lengths[0], self.init_snake_lengths[1])
            # copy action list so we can shuffle them
            shuffled_actions: list = self.all_actions.copy()
            # init snake body list
            snake_body: list = []

            # choose valid position for head
            if len(self.board.open_board_positions) == 0:
                # if there are none kill snake
                self.done = True
                return
            snake_body.append(self.board.random_open_tile_coord())
            self.board.place_tile(self.snake_tiles["snake_head_tile"], snake_body[0])

            # go through each snake body
            for body_count in range(rand_max_length):
                # pick one open position around snake
                shuffle(shuffled_actions)
                for tail_offset in shuffled_actions:
                    tail_coord: array = snake_body[body_count] - tail_offset
                    if not self.board.board_tiles[tail_coord[0]][tail_coord[1]].occupy:
                        # append open position to snake
                        snake_body.append(tail_coord)
                        self.board.place_tile(self.snake_tiles["snake_tile"], tail_coord)
                        break
                else:
                    # break if no positions around snake are open
                    break

            return snake_body, tail_offset

        self.done: bool = False
        self.snake_body, self.snake_direction = create_snake()

    def remove_snake_tail(self) -> array:
        '''
        Removes tail bodypart from snake
        '''
        tail_point: array = self.snake_body.pop()
        self.board.place_tile(AirTile(), tail_point)
        return tail_point

    def remove_snake_head(self) -> array:
        '''
        Removes head bodypart from snake
        '''
        head_point: array = self.snake_body.pop(0)
        self.board.place_tile(AirTile(), head_point)
        return head_point

    def place_new_snake_head(self, snake_coord: array) -> None:
        '''
        Adds new head to snake
        '''
        if (len(self.snake_body) != 0):
            self.board.place_tile(self.snake_tiles["snake_tile"], self.snake_body[0])
        self.snake_body.insert(0, snake_coord)
        self.board.place_tile(self.snake_tiles["snake_head_tile"], snake_coord)
        
    def place_new_snake_tail(self, snake_coord: array) -> None:
        '''
        Adds new bodypart to snake
        '''
        self.snake_body.append(snake_coord)
        self.board.place_tile(self.snake_tiles["snake_tile"], snake_coord)

    def move(self, direction_index: FloatTensor, random_action: FloatTensor) -> FloatTensor:
        '''
        Moves snake in direction and returns reward
        '''

        # set direction if dot product is not negative
        if ((self.all_actions[direction_index] != -self.snake_direction).all()):
            self.snake_direction = self.all_actions[direction_index]

        reward_sum: float = 0

        for _ in range(self.snake_move_count):
            moved_head_point: array = self.snake_body[0] + self.snake_direction

            # move tail
            tail_point: array = self.remove_snake_tail()
            
            # check what tile you hit
            tile: Tiles = self.board.board_tiles[moved_head_point[0]][moved_head_point[1]]

            # move snake head
            self.place_new_snake_head(moved_head_point)
            
            # react to what tile you hit
            reward_sum += tile.on_hit(self, old_snake_tail_pos=tail_point)

            if (self.done):
                self.death += 1
                if (random_action == 1.):
                    self.random_action_death += 1
                break

        return FloatTensor([reward_sum])