# backend/explore.py run: python backend/explore.py data/demos/match01.dem
import sys
from demoparser2 import DemoParser

if len(sys.argv) < 2:
    print("Usage: python backend/explore.py <path_to_demo>")
    sys.exit(1)

p = DemoParser(sys.argv[1])
print(p.parse_header())
for ev in ['round_freeze_end', 'round_end', 'bomb_planted',
           'smokegrenade_detonate', 'player_hurt']:
    d = p.parse_event(ev)
    print('=== event:', ev, '| rows:', len(d))
    print(d.head(3))

fe = p.parse_event('round_freeze_end')
if len(fe) > 0:
    first = int(fe.tick.iloc[0]) + 64
    props = ['X', 'Y', 'yaw', 'health', 'armor_value', 'team_num', 'is_alive',
             'balance', 'active_weapon_name']
    print(p.parse_ticks(props, ticks=[first]))
