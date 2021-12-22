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
        min_board_shape         = array([100, 100]),
        max_board_shape         = array([100, 100]),
        replay_interval         = 0,
        snakes                  = [random_snake],
    )

    step_count: int = 0 
    start_time = time.time()

    while True:
        board.__restart__()

        while board.is_alive():
            action: int = random_snake.act()
            reward: float = random_snake.move(action)

            step_count+=1
            if step_count % 1_000 == 0:
                print('\rAPS: {:.0f}'.format(step_count / (time.time() - start_time)))
