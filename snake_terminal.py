from snake_environment import *
from msvcrt import getch

def KeyCheck() -> array:
    global Break_KeyCheck
    Break_KeyCheck = False
    
    base = getch()
    if base == b'\x00':
        sub = getch()
        
        if sub == b'H':
            return array([-1, 0])

        elif sub == b'M':
            return array([0, 1])

        elif sub == b'P':
            return array([1, 0])

        elif sub == b'K':
            return array([0, -1])

if __name__ == "__main__":


    snake: Snake = Snake(
        Board.BoardData(
            array([3, 3]),
            array([5, 5]),
            0.0,
            1,
            10
        )
    )

    all_rewards: int = 0
    while True:
        snake.__restart__()
        while not snake.done:
            #print(chr(27) + "[2J")
            print(snake.board_data.board)

            player_input: array = KeyCheck()

            this_reward: float = snake.move_snake(player_input)
            all_rewards += this_reward
            print(all_rewards)

        print(snake.board_data.board_replay)
        KeyCheck()
