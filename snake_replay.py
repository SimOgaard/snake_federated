import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import sys
from torch import tensor

from composed_functions.dqn_helper import load_checkpoint

class Replay():
  '''
  Snake game replay
  '''
  def __init__(self, replay_list = [tensor([[0, 0], [0, 0]])]) -> None:

    super(Replay, self).__init__()

    self.replay_list = replay_list;

    for i in range(0, len(self.replay_list)):
      self.replay_list[i] = self.replay_list[i].tolist()

  def replay_snake(self):

    frame = self.replay_list[0]

    def update(data):
      mat.set_data(data)
      return mat

    # create discrete colormap
    cmap = colors.ListedColormap(['royalblue', 'navy', 'darkred', 'limegreen', 'orange', 'yellow'])

    fig, ax = plt.subplots()

    plt.axis('off')

    mat = ax.matshow(frame, cmap=cmap)

    plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, self.replay_list, interval=25)

    plt.show()

  def append_replay(self, frame):

    self.replay_list.append(frame.tolist())


if __name__ == "__main__":
  checkpoint = load_checkpoint(sys.argv[1])

  replay: Replay = Replay(
    replay_list             = checkpoint['replay']
  )

  replay.replay_snake()