from snake_environment import *
from msvcrt import getch

def KeyCheck() -> array:
    global Break_KeyCheck
    Break_KeyCheck = False
    
    base = getch()
    if base == b'\x00':
        sub = getch()
        
        if sub == b'H':
            print("up")
            return array([-1, 0])

        elif sub == b'M':
            print("höger")
            return array([0, 1])

        elif sub == b'P':
            print("ner")
            return array([1, 0])

        elif sub == b'K':
            print("vänster")
            return array([0, -1])

if __name__ == "__main__":
    reward_data: Rewards = Rewards()
    board_data: Board = Board()
    snake: SnakeAgent = SnakeAgent(reward_data, board_data)

    all_rewards: int = 0
    while True:
        print(chr(27) + "[2J")
        print(snake.board_data.board)

        player_input: array = KeyCheck()

        this_reward: float = snake.move_snake(player_input)
        all_rewards += this_reward
        print(all_rewards)

        if (snake.done):
            break