class Tiles():
    '''
    Holds common functionality for tiles on board
    Made to make it easy to add new tiles
    '''

    def __init__(self, visual: int, reward: int, occupy: bool, salt_pepper_chance: float, spawn_amount: int) -> None:
        '''
        Common tiles init (values)
        '''
        super(Tiles, self).__init__()

        self.visual = visual
        self.reward = reward
        self.occupy = occupy
        self.salt_pepper_chance = salt_pepper_chance
        self.spawn_amount = spawn_amount

    def on_hit(self, snake, **kwargs: dict) -> None:
        '''
        Tile on_hit function
        (currently only virtual placeholder, implement ovveride when needed)
        '''
        pass