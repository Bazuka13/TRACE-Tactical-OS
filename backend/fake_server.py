# backend/fake_server.py run: uvicorn fake_server:app --app-dir backend --port 8000
import asyncio
import math
import orjson
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'],
                   allow_headers=['*'])


@app.websocket('/ws/replay/{match_id}/{round_num}')
async def fake(ws: WebSocket, match_id: str, round_num: int):
    await ws.accept()
    for i in range(600):
        players = [{'slot': s, 'team': 0 if s < 5 else 1,
                    'x': 0.5 + 0.3 * math.cos(i / 30 + s),
                    'y': 0.5 + 0.3 * math.sin(i / 30 + s),
                    'yaw': (i * 3 + s * 30) % 360,
                    'hp': 1.0,
                    'alive': True,
                    'threat': (s % 3) / 2,
                    'why': 'fake explanation for player %d' % (s + 1)}
                   for s in range(10)]
        payload = {
            'tick': i,
            'map': 'de_mirage',
            'clock': i / 10,
            'players': players,
            'territory': [[0 if c < 32 else 1 for c in range(64)] for _ in range(64)],
            'danger': [[0.5 if (r < 8 and c < 8) else 0.0 for c in range(32)] for r in range(32)],
            'win_prob': 0.5 + 0.3 * math.sin(i / 50),
            'why_win': 'fake win explanation',
            'bomb_planted': 0
        }
        await ws.send_text(orjson.dumps(payload).decode())
        await asyncio.sleep(0.1)
