import pygame, random
from training_game import Hexasphere, Game

def render(screen, game, width, height):
    screen.fill((10,10,20))
    hexs = game.hex
    rx, ry = game.rot
    scale = 230
    cx, cy = width//2, height//2
    for tid,t in hexs.tiles.items():
        poly2d=[]
        center = hexs.project(t["center"], (rx, ry))
        if center[2] <= 0:
            continue
        poly2d = []
        for p in t["polygon3d"]:
            x, y, z = hexs.project(p, (rx, ry))
            poly2d.append((cx + x*scale, cy - y*scale))
        color = (80,80,80) if game.tiles[tid].owner is None else [(200,50,50),(50,150,250),(0,100,50),(200,50,250)][game.tiles[tid].owner]
        pygame.draw.polygon(screen, color, poly2d, 0)
        pygame.draw.polygon(screen, (0,0,0), poly2d, 1)
    for (aid,pid),piece in game.pieces.items():
        x,y,z = hexs.project(game.tiles[piece.tile_id].center3d,(rx,ry))
        if z>0:
            col = [(255,100,100),(100,180,255),(100,200,100),(255,100,255)][aid]
            r = 6 if (aid,pid)!=game.selected else 9
            pygame.draw.circle(screen, col, (int(cx+x*scale),int(cy-y*scale)), r)
    pygame.display.flip()

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
                elif e.button==3:
                    mx,my=e.pos
                    sel=None
                    for (aid,pid),p in game.pieces.items():
                        x,y,z=hexs.project(game.tiles[p.tile_id].center3d,(game.rot[0],game.rot[1]))
                        if z>0:
                            px,py=W//2+x*230,H//2-y*230
                            if (mx-px)**2+(my-py)**2<100:
                                if aid==game.current_player:
                                    sel=(aid,pid)
                    game.selected=sel
            elif e.type==pygame.MOUSEBUTTONUP and e.button==1:
                dragging=False
            elif e.type==pygame.MOUSEMOTION and dragging:
                dx,dy=e.rel
                game.rot[1]+=dx*0.01
                game.rot[0]+=dy*0.01
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: running=False
                if e.key==pygame.K_SPACE:
                    ai = 1-game.current_player
                    game.step_ai(ai)
                    if game.winner is not None:
                        running = False
                    else:
                        game.current_player = ai if game.current_player==0 else 0
                if e.key==pygame.K_RETURN and game.selected:
                    p = game.pieces[game.selected]
                    moves = game.legal_moves(p)
                    if moves:
                        dest = random.choice(moves)
                        game.move(p, dest)
                        if game.winner is not None:
                            running = False
        render(screen,game,W,H)
        clock.tick(30)
    pygame.quit()

if __name__=="__main__":
    main()
