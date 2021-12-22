# math functions
from numpy import array

# this repo imports
from snake_env.tiles.virtual_tiles import *

class AirTile(Tiles):
    '''
    Tile that does nothing
    '''
    def __init__(self, visual: int = 0, reward: int = -0.01, occupy: bool = False) -> None:
        '''
        Initialize a air tile object.
        '''
        super(AirTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Does nothing
        '''
        return self.reward

class WallTile(Tiles):
    '''
    Tile that represents an wall
    '''
    def __init__(self, visual: int = 1, reward: int = -1., occupy: bool = True) -> None:
        '''
        Initialize a edge tile object.
        '''
        super(WallTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Kills snake
        '''
        snake.done = True
        return self.reward

class MineTile(Tiles):
    '''
    A mine that removes a bodypart from snake
    '''
    def __init__(self, visual: int = 2, reward: int = -0.25, occupy: bool = True) -> None:
        '''
        Initialize a mine tile object.
        '''
        super(MineTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Removes tail from snake 
        Kills snake if snake body length is zero
        '''
        snake.remove_snake_tail()
        if (len(snake.snake_body) == 0):
            snake.done = True
            return -1 # return larger penalty because snake died
        return self.reward

class FoodTile(Tiles):
    '''
    Adds new bodypart to snake
    '''
    def __init__(self, visual: int = 3, reward: int = 1, occupy: bool = True) -> None:
        '''
        Initialize a food tile object.
        '''
        super(FoodTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Adds new bodypart to snake
        '''
        # add snake part
        if ("old_snake_tail_pos" in kwargs):
            snake.place_new_snake_tail(kwargs["old_snake_tail_pos"])

        # place new food on board
        if (len(snake.board.open_board_positions) != 0):
            # get random coord viable for food placement
            random_coord: array = snake.board.random_open_tile_coord()
            snake.board.place_tile(FoodTile(), random_coord)
        return self.reward

class SnakeTile(Tiles):
    '''
    Tile representing a snake bodypart
    '''
    def __init__(self, visual: int = 4, reward: int = -1., occupy: bool = True) -> None:
        '''
        Initialize a snake body tile object.
        '''
        super(SnakeTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Kills snake
        '''
        snake.done = True
        return self.reward

class SnakeHeadTile(SnakeTile):
    '''
    Tile representing a snake head
    '''
    def __init__(self, visual: int = 5) -> None:
        '''
        Initialize a snake head tile object.
        '''
        super(SnakeHeadTile, self).__init__(visual = visual)

class InvertTile(Tiles):
    '''
    Tile that inverts the direction of the snake
    '''
    def __init__(self, visual: int = 6, reward: int = 0., occupy: bool = True) -> None:
        '''
        Initialize a snake head tile object.
        '''
        super(InvertTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Inverts the direction of the snake
        '''
        if len(snake.snake_body) > 1:
            snake.snake_body.reverse()
            snake.snake_direction = -(snake.snake_body[-1] - snake.snake_body[-2])
        else:
            snake.snake_direction = -snake.snake_direction

        # set old snake head position to right tile
        snake.board.place_tile(SnakeTile(), snake.snake_body[-1])
        # set new snake head position to right tile
        snake.board.place_tile(SnakeHeadTile(), snake.snake_body[0])

        return self.reward

# class SplitTile(Tiles):
#     '''
#     Tile that splits the snake in two and creates a new snake from lost bodypart
#     '''
#     def __init__(self, visual: int = 7, reward: int = -1., occupy: bool = True) -> None:
#         '''
#         Initialize a split tile object.
#         '''
#         super(SplitTile, self).__init__(visual, reward, occupy)

#     def on_hit(self, snake, **kwargs: dict) -> float:
#         '''
#         Splits snake in two and creates a new snake from last bodypart
#         '''
#         if len(snake.snake_body) > 1:
#             # split snake
#             middle_index: int = len(snake.snake_body) // 2

#             # create a snake from split list
#             new_snake: RandomAgent = RandomAgent()
#             snake.board.temporary_snakes.append(new_snake)
#             new_snake.board = snake.board
#             new_snake.done = False
#             new_snake.snake_body = snake.snake_body[middle_index:]
#             new_snake.snake_direction = -(new_snake.snake_body[-1] - new_snake.snake_body[-2])

#             snake.snake_body = snake.snake_body[:middle_index]

#             # add head to board
#             snake.board.place_tile(SnakeHeadTile(), new_snake.snake_body[0])

#         return self.reward

