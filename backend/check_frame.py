# backend/check_frame.py run: python backend/check_frame.py match01 3 40
# (match, round number, seconds into the round)
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from config import FEATS, MAPS
from maps import load_walls

if len(sys.argv) < 4:
    print("Usage: python backend/check_frame.py <match_id> <round_num> <sec_into_round>")
    sys.exit(1)

mid, rnd, sec = sys.argv[1].replace('_players', '').replace('.parquet', ''), int(sys.argv[2]), float(sys.argv[3])
pl = pd.read_parquet(FEATS / (mid + '_players.parquet'))
r = pl[pl.round_num == rnd]
if len(r) == 0:
    print(f"Error: Round {rnd} not found in {mid}")
    sys.exit(1)

tick = r.tick.iloc[(r.tick - (r.tick.min() + sec * 64)).abs().argmin()]
f = r[r.tick == tick].sort_values('slot')
map_name = f.map_name.iloc[0]

plt.figure(figsize=(9, 9))
img_path = MAPS / (map_name + '.png')
if img_path.exists():
    plt.imshow(Image.open(img_path), extent=(0, 1, 1, 0))

for a, b in load_walls(map_name):
    plt.plot([a[0], b[0]], [a[1], b[1]], 'y-', lw=1)  # walls in yellow

for p in f.itertuples():
    if not p.is_alive:
        continue
    for j in range(p.slot + 1, 10):
        if (p.los_bits >> j) & 1:
            o_rows = f[f.slot == j]
            if len(o_rows) > 0:
                o = o_rows.iloc[0]
                plt.plot([p.pos_x, o.pos_x], [p.pos_y, o.pos_y], 'g-' if o.team == p.team else 'r-', lw=1)

plt.scatter(f.pos_x, f.pos_y, c=f.team, cmap='coolwarm', s=80, edgecolors='k')
plt.title('green = teammates see each other, red = enemies see each other')
plt.show()
