from random import randrange

def better_rand(x: int, y: int):
    '''
    Workaround this stupid fucking code snippet: assert x != y in random.randrange
    '''
    if (x != y):
        return randrange(x, y)
    return x
