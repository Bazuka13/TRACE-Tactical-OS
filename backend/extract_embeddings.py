# backend/extract_embeddings.py run: python backend/extract_embeddings.py
import numpy as np
import pandas as pd
import torch
from config import EMB, MODELS, GNN_COLS
from dataset import all_match_ids
from gnn_utils import TacticalSTGNN, load_rounds, frame_graph, make_batch, WINDOW

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def embed_round(model, r):
    F = len(r['ticks'])
    S = []
    for a in range(0, F, 256):  # step 1: the graph part, once per frame
        frames = [frame_graph(r['x'][j], r['alive'][j], r['los'][j], r['pos'][j], r['thr'])
                  for j in range(a, min(F, a + 256))]
        x, ei, gid = make_batch([frames])
        S.append(model.spatial(x.to(dev), ei.to(dev), gid.to(dev)).cpu())
    S = torch.cat(S)  # (frames, 192)
    S = torch.cat([S[:1].repeat(WINDOW - 1, 1), S])  # pad the start of the round
    win = S.unfold(0, WINDOW, 1).permute(0, 2, 1)    # (frames, 50, 192): 5 s ending at each frame
    out = [model.temporal(win[a:a + 1024].to(dev)).cpu() for a in range(0, F, 1024)]
    return torch.cat(out).numpy()  # (frames, 128)


if __name__ == '__main__':
    stgnn_file = MODELS / 'stgnn.pt'
    if not stgnn_file.exists():
        print(f"Error: {stgnn_file} not found. Run train_stgnn.py first.")
        sys.exit(1)

    model = TacticalSTGNN().to(dev)
    model.load_state_dict(torch.load(stgnn_file, map_location=dev))
    model.eval()

    for m in all_match_ids():
        parts = []
        try:
            rounds = load_rounds(m)
        except Exception as e:
            print(f"Skipping {m}: {e}")
            continue
        for r in rounds:
            d = pd.DataFrame(embed_round(model, r).astype(np.float32), columns=GNN_COLS)
            d.insert(0, 'tick', r['ticks'])
            d.insert(0, 'round_num', r['round_num'])
            d.insert(0, 'match_id', m)
            parts.append(d)
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(EMB / (m + '_gnn.parquet'))
            print('embedded', m)
