from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

if __name__ == "__main__":
    player_snake: ControllableAgent = ControllableAgent()
    random_snake: RandomAgent = RandomAgent()

    board: Board = Board(
        min_board_shape         = array([3, 3]),
        max_board_shape         = array([5, 5]),
        salt_and_pepper_chance  = 0.0,
        food_amount             = 1,
        replay_interval         = 10,
        snakes                  = [player_snake, random_snake]
    )

    while True:
        for step in board:
            action, observation, reward, done = player_snake.get_step()
        print("Every snake died, restarting environment...")