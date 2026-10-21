# backend/train_stgnn.py run: python backend/train_stgnn.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from config import MODELS
from dataset import match_ids
from gnn_utils import TacticalSTGNN, load_rounds, frame_graph, make_batch, WINDOW

STRIDE = 10  # use every 10th frame as a sample (neighbouring frames are near-copies)
BATCH = 16
EPOCHS = 12


class WindowDS(Dataset):
    def __init__(self, rounds, stride):
        self.rounds = rounds
        self.items = [(i, f) for i, r in enumerate(rounds) for f in range(0, len(r['ticks']), stride)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, f = self.items[k]
        r = self.rounds[i]
        idxs = np.clip(np.arange(f - WINDOW + 1, f + 1), 0, None)  # last 50 frames (padded at round start)
        frames = [frame_graph(r['x'][j], r['alive'][j], r['los'][j], r['pos'][j], r['thr']) for j in idxs]
        return frames, r['y']


def collate(batch):
    x, ei, gid = make_batch([b[0] for b in batch])
    return x, ei, gid, torch.tensor([b[1] for b in batch], dtype=torch.float32), len(batch)


def run_epoch(model, dl, dev, opt=None):
    train = opt is not None
    model.train(train)
    losses, ys, ps = [], [], []
    for x, ei, gid, y, B in dl:
        x, ei, gid, y = x.to(dev), ei.to(dev), gid.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            logit, _ = model(x, ei, gid, B, WINDOW)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
        ys += y.tolist()
        ps += torch.sigmoid(logit).tolist()
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else 0.5
    return float(np.mean(losses)), auc


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)
    tr_r = [r for m in match_ids('train') for r in load_rounds(m)]
    va_r = [r for m in match_ids('val') for r in load_rounds(m)]
    if not tr_r or not va_r:
        print("Error: train or val rounds missing. Please ingest data first.")
        sys.exit(1)

    tr = DataLoader(WindowDS(tr_r, STRIDE), BATCH, shuffle=True, collate_fn=collate)
    va = DataLoader(WindowDS(va_r, STRIDE * 2), BATCH, collate_fn=collate)
    model = TacticalSTGNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = 9.0
    for ep in range(EPOCHS):
        tl, ta = run_epoch(model, tr, dev, opt)
        vl, vauc = run_epoch(model, va, dev)
        print('epoch %d train loss %.3f auc %.3f | val loss %.3f auc %.3f' % (
            ep + 1, tl, ta, vl, vauc))
        if vl < best:
            best = vl
            torch.save(model.state_dict(), MODELS / 'stgnn.pt')  # keep the best version
