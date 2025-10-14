import random

class Tile:
    # the tiles currently just contain info on ownership and resources. More can be added later
    def __init__(self, tid, center):
        self.id = tid
        self.center3d = center
        self.owner = None
        self.resources = random.randint(1,5)
        self.piece = None