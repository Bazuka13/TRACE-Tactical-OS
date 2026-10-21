# backend/universal_tick.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class PlayerTick:
    player_id: int
    team: int  # 0 = Attack (T), 1 = Defense (CT)
    pos_x: float  # 0..1 on the radar
    pos_y: float
    yaw: float  # degrees, in RADAR space
    hp_pct: float  # 0..1
    armor_pct: float  # 0..1
    weapon_dps: float  # firepower rating 0..1
    is_alive: bool
    vel_x: float = 0.0  # filled in Phase 1
    vel_y: float = 0.0


@dataclass
class UniversalTickPacket:
    game: str  # 'cs2' or 'valorant'
    match_id: str
    round_num: int
    tick: int
    round_clock_s: float
    team_power_idx: float  # attack team power, 0..1
    defender_power_idx: float = 0.0
    players: List[PlayerTick] = field(default_factory=list)
    smoke_centers: list = field(default_factory=list)  # [(x, y, radius)]


def df_to_packets(df):
    # turns a Phase 0 table into a stream of packets (one per frame)
    for (rnd, tick), g in df.sort_values(['round_num', 'tick', 'slot']).groupby(['round_num', 'tick']):
        att_rows = g[g.team == 0]
        dfn_rows = g[g.team == 1]
        att = att_rows.team_power_idx.iloc[0] if len(att_rows) else 0.0
        dfn = dfn_rows.team_power_idx.iloc[0] if len(dfn_rows) else 0.0
        yield UniversalTickPacket(
            game='cs2', match_id=g.match_id.iloc[0], round_num=int(rnd),
            tick=int(tick),
            round_clock_s=float(g.round_clock_s.iloc[0]),
            team_power_idx=float(att),
            defender_power_idx=float(dfn),
            players=[PlayerTick(int(r.player_id), int(r.team), float(r.pos_x),
                                float(r.pos_y), float(r.yaw), float(r.hp_pct),
                                float(r.armor_pct), float(r.weapon_dps), bool(r.is_alive))
                     for r in g.itertuples()])
