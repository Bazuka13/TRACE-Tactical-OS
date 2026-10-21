# backend/draw_walls.py run: python backend/draw_walls.py de_mirage
import sys
import json
import matplotlib.pyplot as plt
from PIL import Image
from config import MAPS

if len(sys.argv) < 2:
    print("Usage: python backend/draw_walls.py <map_name>")
    sys.exit(1)

name = sys.argv[1]
out = MAPS / (name + '_walls.json')
segs = json.load(open(out)) if out.exists() else []

fig, ax = plt.subplots(figsize=(10, 10))
img_path = MAPS / (name + '.png')
if img_path.exists():
    ax.imshow(Image.open(img_path), extent=(0, 1, 1, 0))
else:
    print(f"Warning: map image not found at {img_path}. Proceeding with blank canvas.")

for a, b in segs:
    ax.plot([a[0], b[0]], [a[1], b[1]], 'r-')  # walls you already drew

ax.set_title('Click START then END of each wall. Press ENTER when finished.')
pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
for i in range(0, len(pts) - 1, 2):
    segs.append([list(pts[i]), list(pts[i + 1])])

with open(out, 'w') as f:
    json.dump(segs, f, indent=2)
print('saved', len(segs), 'walls to', out)
