# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

# Generic imports
import time

if __name__ == "__main__":
    '''
    Creates a large environment and counts achived apm of random agent
    '''
    random_snake: RandomAgent = RandomAgent(init_snake_lengths=array([2, 10]))

    board: Board = Board(
        min_board_shape         = array([500, 500]),
        max_board_shape         = array([500, 500]),
        replay_interval         = 0,
        snakes                  = [random_snake],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(),
            "mine_tile": MineTile()
        },
    )

    step_count: int = 0
    board_create_count: int = 0
    start_time = time.time()

    while True:
        # import time
        # start = time.time()

        board_create_count+=1
        board.__restart__()

        # while board.is_alive():
        #     action, is_random = random_snake.act()
        #     reward: float = random_snake.move(action, is_random)

        #     step_count+=1
        #     if step_count % 100 == 0:
        #         print('\rAPS: {:.0f}'.format(step_count / (time.time() - start_time)))

        # end = time.time()
        # print(end - start)

        if board_create_count % 5 == 0:
            print('\rAPS: {:.5f}'.format(board_create_count / (time.time() - start_time)))
