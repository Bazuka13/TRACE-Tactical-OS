# backend/maps.py
import json
import numpy as np
from config import MAPS

RADAR_SIZE = 1024  # radar images are 1024 x 1024 pixels

# pos_x, pos_y = world position of the radar image top-left corner; scale = world units per pixel.
# These are the standard values used by the awpy library. VERIFY them with check_map.py below.
MAP_DATA = {
    'de_mirage': {'pos_x': -3230, 'pos_y': 1713, 'scale': 5.0},
    'de_dust2': {'pos_x': -2476, 'pos_y': 3239, 'scale': 4.4},
    'de_inferno': {'pos_x': -2087, 'pos_y': 3870, 'scale': 4.9},
    'de_overpass': {'pos_x': -4831, 'pos_y': 1781, 'scale': 5.2},
    'de_ancient': {'pos_x': -2953, 'pos_y': 2164, 'scale': 4.2},
    'de_anubis': {'pos_x': -2796, 'pos_y': 3328, 'scale': 5.22},
    'de_nuke': {'pos_x': -3453, 'pos_y': 2887, 'scale': 7.0},
}


def world_to_norm(x, y, map_name):
    # game coordinates -> (0..1, 0..1) on the radar image. Y is flipped because images grow downward.
    m = MAP_DATA[map_name]
    px = (np.asarray(x, dtype=float) - m['pos_x']) / m['scale']
    py = (m['pos_y'] - np.asarray(y, dtype=float)) / m['scale']
    return px / RADAR_SIZE, py / RADAR_SIZE


def units_to_norm(units, map_name):
    # a distance in game units -> the same distance on the 0..1 scale
    return units / (MAP_DATA[map_name]['scale'] * RADAR_SIZE)


def units_per_norm(map_name):
    return MAP_DATA[map_name]['scale'] * RADAR_SIZE


def load_walls(map_name):
    # walls = list of line segments [[x1,y1],[x2,y2]] in 0..1 coordinates
    path = MAPS / (map_name + '_walls.json')
    if not path.exists():
        print('WARNING: no wall file for', map_name, '-> everyone will see everyone')
        return np.zeros((0, 2, 2))
    with open(path) as f:
        return np.array(json.load(f), dtype=float).reshape(-1, 2, 2)
