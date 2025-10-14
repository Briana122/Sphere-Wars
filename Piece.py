class Piece:
    # pieces are very simple, just track which agent and tile they are on. Logic for creating more pieces and moving existing pieces exists in the Game class
    def __init__(self, agent, pid, tile):
        self.agent, self.pid, self.tile_id = agent, pid, tile