# backend/gnn_utils.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch_geometric.nn import GATConv, global_mean_pool
from config import FEATS
from maps import units_to_norm

WINDOW = 50  # 50 frames, about 5 seconds

# the plan has 9 inputs; we add 'team' so the network can tell friends from enemies
NODE_FEATS = ['pos_x', 'pos_y', 'yaw_sin', 'yaw_cos', 'hp_pct', 'armor_pct',
              'weapon_dps', 'speed', 'degree', 'team']


class TacticalSTGNN(nn.Module):
    def __init__(self, in_dim=10):
        super().__init__()
        self.gat1 = GATConv(in_dim, 64, heads=3, concat=True)  # 10 -> 192
        self.gat2 = GATConv(192, 64, heads=3, concat=True)      # 192 -> 192
        self.gru = nn.GRU(192, 128, batch_first=True)
        self.head = nn.Linear(128, 1)  # only used while training

    def spatial(self, x, edge_index, batch):
        h = Fn.relu(self.gat1(x, edge_index))
        h = Fn.relu(self.gat2(h, edge_index))
        return global_mean_pool(h, batch)  # one 192-vector per frame

    def temporal(self, seq):  # seq: (batch, 50, 192)
        _, hidden = self.gru(seq)
        return hidden[-1]  # (batch, 128)

    def forward(self, x, edge_index, batch, B, T):
        s = self.spatial(x, edge_index, batch).view(B, T, -1)
        emb = self.temporal(s)
        return self.head(emb).squeeze(-1), emb


def split_rounds(pl, thr):
    # players table of one match -> list of rounds, each with arrays shaped (frames, 10 players, ...)
    rounds = []
    pl = pl.sort_values(['round_num', 'tick', 'slot'])
    for rn, g in pl.groupby('round_num', sort=True):
        F = len(g) // 10
        x = g[NODE_FEATS].to_numpy(np.float32).reshape(F, 10, -1).copy()
        x[..., 7] /= 300.0  # speed in units/s -> about 0..1
        x[..., 8] /= 4.0    # degree 0..4 -> 0..1
        bits = g.los_bits.to_numpy(np.int64).reshape(F, 10)
        rounds.append(dict(
            ticks=g.tick.to_numpy()[::10],
            x=x,
            round_num=int(rn),
            thr=thr,
            pos=g[['pos_x', 'pos_y']].to_numpy(np.float32).reshape(F, 10, 2),
            alive=g.is_alive.to_numpy(bool).reshape(F, 10),
            los=((bits[:, :, None] >> np.arange(10)) & 1).astype(bool),
            y=float(g.round_winner.iloc[0] == 0)  # 1 = attackers win
        ))
    return rounds


def load_rounds(match_id):
    pl = pd.read_parquet(FEATS / (match_id + '_players.parquet'))
    rs = split_rounds(pl, units_to_norm(800, pl.map_name.iloc[0]))
    for r in rs:
        r['match_id'] = match_id
    return rs


def frame_graph(x, alive, los, pos, thr):
    # nodes = living players; edge = close enough AND can see each other
    idx = np.flatnonzero(alive)
    if len(idx) == 0:
        idx = np.array([0])
    remap = -np.ones(10, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    src, dst = np.nonzero(los & (D <= thr) & alive[:, None] & alive[None, :])
    return x[idx], np.stack([remap[src], remap[dst]])


def make_batch(windows):
    # glue many small graphs into one big disconnected graph so the GPU/CPU does one pass
    xs, eis, gid, off, g = [], [], [], 0, 0
    for frames in windows:
        for x, ei in frames:
            xs.append(x)
            eis.append(ei + off)
            gid.append(np.full(len(x), g))
            off += len(x)
            g += 1
    return (
        torch.from_numpy(np.concatenate(xs)),
        torch.from_numpy(np.concatenate(eis, axis=1)).long(),
        torch.from_numpy(np.concatenate(gid)).long()
    )
