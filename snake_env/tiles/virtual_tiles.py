class Tiles():
    '''
    Holds common functionality for tiles on board
    Made to make it easy to add new tiles
    '''

    def __init__(self, visual: int, reward: int, occupy: bool) -> None:
        '''
        Common tiles init (values)
        '''
        super(Tiles, self).__init__()

        self.visual = visual
        self.reward = reward
        self.occupy = occupy

    def on_hit(self, snake, **kwargs) -> None:
        '''
        Tile on_hit function
        (currently only virtual placeholder, implement ovveride when needed)
        '''
        pass