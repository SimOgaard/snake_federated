# Math modules
from random import randrange, shuffle

# Repo imports
from snake_env.tiles.tiles import *

class Snake():
    '''
    Snake (self explanatory i guess)
    '''

    def __init__(self) -> None:
        '''
        Initilizes our snake at given board
        '''
        super(Snake, self).__init__()

        self.snake_move_count: int = 1
        self.all_actions: list = [array([1, 0]), array([-1, 0]), array([0, 1]), array([0, -1])]
        self.board = None
        
        # needs to run restart but you do it in externally
        # self.__restart__()

    def __restart__(self) -> None:
        '''
        Places our snake on board and restarts every value
        '''

        def create_snake() -> tuple:
            '''
            Chooses random point inside board_shape and places snake at that point in random direction
            Returns snake body list and direction
            '''
            start_coord: array
            tail_coord: array

            while True:
                start_coord = array([randrange(self.board.bounding_box[0], self.board.bounding_box[2]), randrange(self.board.bounding_box[1], self.board.bounding_box[3])])
                if type(self.board.board_tiles[start_coord[0]][start_coord[1]]) == AirTile:
                    shuffle(self.all_actions)
                    for tail_offset in self.all_actions:
                        tail_coord = start_coord - tail_offset
                        if type(self.board.board_tiles[tail_coord[0]][tail_coord[1]]) == AirTile:
                            break
                    else:
                        continue
                    return [start_coord, tail_coord], tail_offset

        self.done: bool = False
        self.snake_body, self.snake_direction = create_snake()

    def remove_snake_tail(self) -> array:
        tail_point: array = self.snake_body.pop()
        self.board.place_tile(AirTile(), tail_point)
        return tail_point

    def place_new_snake_head(self, snake_coord: array) -> None:
        self.board.place_tile(SnakeTile(), self.snake_body[0])
        self.snake_body.insert(0, snake_coord)
        self.board.place_tile(SnakeHeadTile(), snake_coord)
        
    def place_new_snake_tail(self, snake_coord: array) -> None:
        self.snake_body.append(snake_coord)
        self.board.place_tile(SnakeTile(), snake_coord)

    def move(self, direction_index: int) -> float:
        # set direction if dot product is not negative
        if ((self.all_actions[direction_index] != -self.snake_direction).all()):
            self.snake_direction = self.all_actions[direction_index]

        reward_sum: float = 0

        for i in range(self.snake_move_count):
            moved_head_point: array = self.snake_body[0] + self.snake_direction

            # move tail
            tail_point: array = self.remove_snake_tail()
            
            # check what tile you hit
            tile: Tiles = self.board.board_tiles[moved_head_point[0]][moved_head_point[1]]

            # move snake head
            self.place_new_snake_head(moved_head_point)
            
            # react to what tile you hit
            tile.on_hit(self, old_snake_tail_pos=tail_point)
            reward_sum += tile.reward

        return reward_sum