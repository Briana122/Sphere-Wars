class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.tiles_owned = []
        self.num_tiles_owned = 0
        self.pieces_owned = []
        self.num_pieces_owned = 0
        self.acqumilated_resources = 0
        self.total_moves = 0
    
    def add_tile(self,tile_id: int):
        self.tiles_owned.append(tile_id)
        self.num_tiles_owned += 1

    def add_piece(self,piece_id:int):
        self.pieces_owned.append(piece_id)
        self.num_pieces_owned += 1

    def move(self):
        self.total_moves += 1

    def get_resources(self):
        return self.acqumilated_resources
    
    def set_resources(self,resource):
        self.acqumilated_resources = resource

    
        