from numpy.core.arrayprint import printoptions
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

if __name__ == "__main__":
    dqn_snake: DQNAgent = DQNAgent(
        policy_net = DQNLin(5, 4).to(device),
        target_net = DQNLin(5, 4).to(device)
    )

    board: Board = Board(
        min_board_shape         = array([5, 5]),
        max_board_shape         = array([5, 5]),
        salt_and_pepper_chance  = 0.0,
        food_amount             = 10,
        replay_interval         = 1000,
        snakes                  = [dqn_snake]
    )

    sum_length = 0

    while True:
        for step in board:
            action, observation, reward, done = dqn_snake.get_step()
            reward = tensor([reward], device=device)
            action = tensor([action], device=device)
            dqn_snake.memory.push(observation, action, dqn_snake.observate(), reward)
            dqn_snake.optimize_model()

            #print(observation)

            if board.run % dqn_snake.target_update == 0:
                dqn_snake.target_net.load_state_dict(dqn_snake.policy_net.state_dict())

        sum_length += len(dqn_snake.snake_body)

        if board.run % 100 == 0:
            print(sum_length / (board.run + 1))