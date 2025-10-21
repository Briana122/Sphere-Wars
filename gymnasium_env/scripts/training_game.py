import math, numpy as np
from collections import defaultdict

from game.Game import Game
from gymnasium_env.envs.grid_world_env import GridWorldEnv

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
