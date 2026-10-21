# backend/check_map.py run: python backend/check_map.py match01
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from config import TICKS, MAPS

if len(sys.argv) < 2:
    print("Usage: python backend/check_map.py <match_id>")
    sys.exit(1)

match_id = sys.argv[1].replace('.parquet', '')
df = pd.read_parquet(TICKS / (match_id + '.parquet'))
img_path = MAPS / (df.map_name.iloc[0] + '.png')
if not img_path.exists():
    print(f"Warning: map image not found at {img_path}")
else:
    img = Image.open(img_path)
    plt.imshow(img, extent=(0, 1, 1, 0))  # y grows downward, like the radar

s = df.sample(min(len(df), 20000), random_state=0)
plt.scatter(s.pos_x, s.pos_y, s=1, c=s.team, cmap='coolwarm', alpha=0.4)
plt.title('Player positions over the radar: dots must follow the corridors')
plt.show()
