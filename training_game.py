import math, random, numpy as np
from collections import defaultdict


# core hexasphere generation
# also contains util functions

def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def icosahedron():
    # the core icosahedron the hexasphere is based off of - each subdivision addes hexes filling out the space between the pentagons
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [
        (-1,  t,  0), ( 1,  t,  0), (-1, -t,  0), ( 1, -t,  0),
        ( 0, -1,  t), ( 0,  1,  t), ( 0, -1, -t), ( 0,  1, -t),
        ( t,  0, -1), ( t,  0,  1), (-t,  0, -1), (-t,  0,  1)
    ]
    verts = [normalize(v) for v in verts]
    faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)
    ]
    return verts, faces

def midpoint(a,b): return normalize((np.array(a)+np.array(b))/2)

def subdivide(verts, faces):
    # subdivision function to add more hexes. Don't subdivide past 3 or 4, it gets exponentially harder to run
    edge_mid = {}
    def mid(i,j):
        key = tuple(sorted((i,j)))
        if key in edge_mid: return edge_mid[key]
        m = midpoint(verts[i], verts[j])
        idx = len(verts)
        verts.append(m.tolist())
        edge_mid[key] = idx
        return idx
    new_faces = []
    for (i,j,k) in faces:
        a,b,c = mid(i,j), mid(j,k), mid(k,i)
        new_faces += [(i,a,c),(a,j,b),(c,b,k),(a,b,c)]
    return verts, new_faces

def build_geodesic(subdiv=1):
    verts, faces = icosahedron()
    verts = [list(v) for v in verts]
    for _ in range(subdiv): verts, faces = subdivide(verts, faces)
    verts = [normalize(v) for v in verts]
    return verts, faces

class Hexasphere:
    # contains logic for generating and projecting the hexasphere into 3D space. Also stores neighbor info for tiles
    # for training purposes, can be run without visualization. We'll need to meet to decided how to best encode the state of data for RL, but this is a start
    def __init__(self, subdiv=1):
        verts, faces = build_geodesic(subdiv)
        self.verts = np.array(verts)
        self.faces = [tuple(f) for f in faces]
        self.face_centroids = [normalize(np.mean([self.verts[i] for i in f],axis=0)) for f in self.faces]
        self.vertex_to_faces = defaultdict(list)
        for fi,f in enumerate(self.faces):
            for v in f: self.vertex_to_faces[v].append(fi)
        self.tiles = {}
        neigh = defaultdict(set)
        for a,b,c in self.faces:
            neigh[a].update([b,c])
            neigh[b].update([a,c])
            neigh[c].update([a,b])
        for vi, fids in self.vertex_to_faces.items():
            center = self.verts[vi]
            centroids = [self.face_centroids[f] for f in fids]
            z = center / np.linalg.norm(center)
            x_axis = normalize(np.cross([0,1,0], z)) if abs(z[1]) < 0.9 else normalize(np.cross([1,0,0], z))
            y_axis = np.cross(z, x_axis)
            angles = []
            for c in centroids:
                v = c - np.dot(c,z)*z
                ax,ay = np.dot(v,x_axis), np.dot(v,y_axis)
                angles.append(math.atan2(ay,ax))
            sorted_c = [c for _,c in sorted(zip(angles,centroids))]
            self.tiles[vi] = {"center":center,"polygon3d":sorted_c,"neighbors":list(neigh[vi])}
    def project(self, p, rot):
        rx, ry = rot
        # rotate around y then x
        Ry = np.array([[math.cos(ry),0,math.sin(ry)],
                       [0,1,0],
                       [-math.sin(ry),0,math.cos(ry)]])
        Rx = np.array([[1,0,0],
                       [0,math.cos(rx),-math.sin(rx)],
                       [0,math.sin(rx),math.cos(rx)]])
        x,y,z = (Rx @ Ry @ p)
        return np.array([x,y,z])

# core game logic

class Tile:
    # the tiles currently just contain info on ownership and resources. More can be added later
    def __init__(self, tid, center):
        self.id = tid
        self.center3d = center
        self.owner = None
        self.resources = random.randint(1,5)
        self.piece = None

class Piece:
    # pieces are very simple, just track which agent and tile they are on. Logic for creating more pieces and moving existing pieces exists in the Game class
    def __init__(self, agent, pid, tile):
        self.agent, self.pid, self.tile_id = agent, pid, tile

class Game:
    # the main game class, contains all logic for turns, moving pieces, spawning pieces, and checking victory conditions
    # also stores a log of state transitions for RL training purposes   
    # can be scaled up for more players very easily, just need to implement proper turn ordering
    def __init__(self, hexs, players=2, pieces_per=1):
        self.hex, self.players = hexs, players
        self.tiles = {vi: Tile(vi,t["center"]) for vi,t in hexs.tiles.items()}
        self.resources = {p:0 for p in range(players)}
        self.pieces = {}
        for p in range(players):
            for pid in range(pieces_per):
                free = [t for t in self.tiles if self.tiles[t].piece is None]
                start = random.choice(free)
                piece = Piece(p,pid,start)
                self.pieces[(p,pid)] = piece
                self.tiles[start].piece = (p,pid)
                self.tiles[start].owner = p
                self.resources[p]+=self.tiles[start].resources
        self.current_player = 0
        self.selected = None
        self.rot = [0,0]
        self.winner = None
        self.episode_log = []  # store transitions for RL

    def legal_moves(self, piece):
        # return list of legal tile ids the piece can move to, will need to be fine tuned
        # currently can only move to tile already owned or unoccupied tiles
        current_id = piece.tile_id
        neighbors = self.hex.tiles[current_id]["neighbors"]
        legal = [current_id]
        for n in neighbors:
            tile = self.tiles[n]
            if tile.owner is None or tile.owner == piece.agent:
                legal.append(n)
        return legal

    def move(self, piece, dest):
        old = self.tiles[piece.tile_id]
        old.piece = None
        piece.tile_id = dest
        target = self.tiles[dest]
        if target.piece and target.piece[0] != piece.agent:
            del self.pieces[target.piece]
        target.piece = (piece.agent, piece.pid)
        if target.owner != piece.agent:
            target.owner = piece.agent
            self.resources[piece.agent] += target.resources
            target.resources = 0
            self.check_victory(piece.agent)
        # store state transition for RL
        self.episode_log.append({
            "agent": piece.agent,
            "action": dest,
            "resources": dict(self.resources),
            "winner": self.winner
        })

    def step_ai(self, agent):
        if self.winner is not None: return
        ai_pieces = [x for x in self.pieces.values() if x.agent == agent]
        for piece in ai_pieces:
            moves = self.legal_moves(piece)
            if moves:
                dest = random.choice(moves)
                self.move(piece, dest)
                if self.winner is not None:
                    return
        spawn_cost = 10
        while self.resources[agent] >= spawn_cost:
            spawned = self.spawn_piece(agent, spawn_cost)
            if not spawned:
                break
            if self.winner is not None:
                return

    def spawn_piece(self, agent, cost):
        # some basic logic for spawning new pieces, can only spawn on owned tiles that are unoccupied
        owned_free = [t for t in self.tiles.values() if t.owner == agent and t.piece is None]
        if not owned_free:
            return False
        tile = random.choice(owned_free)
        existing = [pid for (a, pid) in self.pieces if a == agent]
        pid = max(existing, default=-1) + 1
        new_piece = Piece(agent, pid, tile.id)
        self.pieces[(agent, pid)] = new_piece
        tile.piece = (agent, pid)
        self.resources[agent] -= cost
        self.check_victory(agent)
        return True

    def check_victory(self, last_agent=None):
        # basic victory check of if the agent owns half or more of the tiles
        total = len(self.tiles)
        half = total / 2.0
        counts = {p: 0 for p in range(self.players)}
        for t in self.tiles.values():
            if t.owner is not None:
                counts[t.owner] += 1
        for p, cnt in counts.items():
            if cnt >= half:
                self.winner = p
                return True
        return False
