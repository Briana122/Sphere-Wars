import pygame, random
import math
from .Game import Game
from .Piece import Piece
from .Tile import Tile
from ..utils.training_game import Hexasphere

def render(screen, game, width, height, rx=0, ry=0, zoom=1.0):
    screen.fill((10, 10, 20))
    hexs = game.hex
    base_scale = 230    
    scale = base_scale * zoom 
    cx, cy = width // 2, height // 2

    for tid, t in hexs.tiles.items():
        center = hexs.project(t["center"], (rx, ry))
        if center[2] <= 0:
            continue

        poly2d = [
            (cx + hexs.project(p, (rx, ry))[0] * scale,
             cy - hexs.project(p, (rx, ry))[1] * scale)
            for p in t["polygon3d"]
        ]
        
        owner = game.tiles[tid].owner
        base_color = (80, 80, 80) if owner is None else [(200, 50, 50), (50, 150, 250), (0, 100, 50), (200, 50, 250)][owner]

        res = game.tiles[tid].resources
        res_factor = min(max(res * 0.12, -1), 1)  
        # 0.01 = change per resource unit, clamp to [-1, 1]

        color = tuple(
            max(0, min(255, int(c * (1 + res_factor))))
            for c in base_color
        )

        pygame.draw.polygon(screen, color, poly2d, 0)
        pygame.draw.polygon(screen, (0, 0, 0), poly2d, 1)

    for (aid, pid), piece in game.pieces.items():
        x, y, z = hexs.project(game.tiles[piece.tile_id].center3d, (rx, ry))
        if z > 0:
            col = [(255, 100, 100), (100, 180, 255), (100, 200, 100), (255, 100, 255)][aid]
            r = 6 if (aid, pid) != game.selected else 9
            pygame.draw.circle(screen, col, (int(cx + x * scale), int(cy - y * scale)), r)


def main():
    pygame.init()
    W,H=800,800
    screen=pygame.display.set_mode((W,H))
    clock=pygame.time.Clock()
    hexs = Hexasphere(subdiv=3)
    game = Game(hexs)
    running=True
    dragging=False
    last_pos=None

    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.MOUSEBUTTONDOWN:
                if e.button==1:
                    dragging=True
                    last_pos=e.pos