import { useEffect, useRef, useState, type MouseEvent } from 'react';

type P = {
  slot: number;
  team: number;
  x: number;
  y: number;
  yaw: number;
  hp: number;
  alive: boolean;
  threat: number;
  why: string;
};

type Frame = {
  tick: number;
  map: string;
  clock: number;
  players: P[];
  territory: number[][];
  danger: number[][];
  win_prob: number;
  why_win: string;
  bomb_planted: number;
};

const SIZE = 800;
const WS = 'ws://localhost:8000/ws/replay/';
const threatColor = (t: number) => (t > 0.7 ? '#ef4444' : t > 0.3 ? '#f59e0b' : '#22c55e');

function draw(
  ctx: CanvasRenderingContext2D,
  f: Frame | null,
  img: HTMLImageElement | null,
  hover: number | null
) {
  ctx.clearRect(0, 0, SIZE, SIZE);

  // 1. Radar image
  if (img && img.complete) {
    ctx.drawImage(img, 0, 0, SIZE, SIZE);
  } else {
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, SIZE, SIZE);
  }

  if (!f) return;

  // 2. Territory colours (Voronoi 64x64)
  if (f.territory && f.territory.length > 0) {
    const tc = SIZE / f.territory.length;
    f.territory.forEach((row, r) =>
      row.forEach((v, c) => {
        if (v < 0) return;
        ctx.fillStyle = v === 0 ? 'rgba(249,115,22,0.22)' : 'rgba(0,229,255,0.22)';
        ctx.fillRect(c * tc, r * tc, tc + 0.5, tc + 0.5);
      })
    );
  }

  // 3. Danger zones (32x32)
  if (f.danger && f.danger.length > 0) {
    const dc = SIZE / f.danger.length;
    f.danger.forEach((row, r) =>
      row.forEach((v, c) => {
        if (v < 0.3) return;
        ctx.fillStyle = 'rgba(239,68,68,' + v * 0.45 + ')';
        ctx.fillRect(c * dc, r * dc, dc + 0.5, dc + 0.5);
      })
    );
  }

  // 4. Players
  const now = performance.now();
  for (const p of f.players) {
    if (!p.alive) continue;
    const x = p.x * SIZE;
    const y = p.y * SIZE;
    const a = (p.yaw * Math.PI) / 180;

    // Facing cone (90 degrees)
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, 60, a - Math.PI / 4, a + Math.PI / 4);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.fill();

    // Threat ring (pulsing when threat > 0.7)
    const pulse = p.threat > 0.7 ? 3 * Math.sin(now / 150) : 0;
    ctx.beginPath();
    ctx.arc(x, y, 14 + pulse, 0, 2 * Math.PI);
    ctx.strokeStyle = threatColor(p.threat);
    ctx.lineWidth = 3;
    ctx.stroke();

    // The dot, dimmer when low hp
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.globalAlpha = 0.35 + 0.65 * p.hp;
    ctx.fillStyle = p.team === 0 ? '#f97316' : '#00e5ff';
    ctx.fill();
    ctx.globalAlpha = 1;

    // Hover ring highlight
    if (hover === p.slot) {
      ctx.beginPath();
      ctx.arc(x, y, 18, 0, 2 * Math.PI);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

export default function App() {
  const [matchId, setMatchId] = useState('match01');
  const [round, setRound] = useState(1);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [status, setStatus] = useState('idle');

  const canvas = useRef<HTMLCanvasElement>(null);
  const latest = useRef<Frame | null>(null);
  const hoverRef = useRef<number | null>(null);
  const img = useRef<HTMLImageElement | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    let id = 0;
    const loop = () => {
      const ctx = canvas.current?.getContext('2d');
      if (ctx) draw(ctx, latest.current, img.current, hoverRef.current);
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, []);

  const start = () => {
    ws.current?.close();
    const s = new WebSocket(WS + matchId + '/' + round);
    ws.current = s;
    setStatus('connecting');

    s.onopen = () => setStatus('playing');
    s.onclose = () => setStatus('finished');
    s.onerror = () => setStatus('connection error');

    s.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.error) {
        setStatus('error: ' + d.error);
        return;
      }
      if (d.done) {
        setStatus('round ended');
        return;
      }
      if (!img.current || img.current.dataset.map !== d.map) {
        const im = new Image();
        im.src = '/maps/' + d.map + '.png';
        im.dataset.map = d.map;
        img.current = im;
      }
      latest.current = d;
      setFrame(d);
    };
  };

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - r.left) / r.width) * SIZE;
    const my = ((e.clientY - r.top) / r.height) * SIZE;
    let best: number | null = null;
    let bd = 25;
    for (const p of latest.current?.players ?? []) {
      if (!p.alive) continue;
      const d = Math.hypot(p.x * SIZE - mx, p.y * SIZE - my);
      if (d < bd) {
        bd = d;
        best = p.slot;
      }
    }
    hoverRef.current = best;
    setHover(best);
  };

  const win = frame ? Math.round(frame.win_prob * 100) : 50;
  const hovered = frame?.players.find((p) => p.slot === hover);
  const alive = (frame?.players ?? []).filter((p) => p.alive).sort((a, b) => b.threat - a.threat);

  return (
    <div
      style={{
        display: 'flex',
        gap: 24,
        padding: 24,
        fontFamily: "'Inter', -apple-system, sans-serif",
        background: '#0f172a',
        color: '#e2e8f0',
        minHeight: '100vh',
      }}
    >
      <div>
        <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 600, color: '#38bdf8' }}>Match:</span>
          <input
            value={matchId}
            onChange={(e) => setMatchId(e.target.value)}
            style={{
              padding: '6px 10px',
              borderRadius: 6,
              border: '1px solid #334155',
              background: '#1e293b',
              color: '#f8fafc',
            }}
          />
          <span style={{ fontWeight: 600, color: '#38bdf8', marginLeft: 8 }}>Round:</span>
          <input
            type="number"
            min={1}
            value={round}
            onChange={(e) => setRound(+e.target.value)}
            style={{
              width: 60,
              padding: '6px 10px',
              borderRadius: 6,
              border: '1px solid #334155',
              background: '#1e293b',
              color: '#f8fafc',
            }}
          />
          <button
            onClick={start}
            style={{
              padding: '6px 16px',
              borderRadius: 6,
              border: 'none',
              background: '#2563eb',
              color: '#fff',
              fontWeight: 600,
              cursor: 'pointer',
              marginLeft: 8,
            }}
          >
            Play round
          </button>
          <span style={{ marginLeft: 12, fontSize: 13, color: '#94a3b8' }}>[{status}]</span>
        </div>

        {/* Win Probability Bar */}
        <div
          style={{
            display: 'flex',
            height: 28,
            borderRadius: 6,
            overflow: 'hidden',
            marginBottom: 12,
            lineHeight: '28px',
            fontWeight: 600,
            fontSize: 13,
            boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
          }}
        >
          <div
            style={{
              width: win + '%',
              background: '#f97316',
              paddingLeft: 10,
              whiteSpace: 'nowrap',
              transition: 'width 0.2s ease',
            }}
          >
            Attack {win}%
          </div>
          <div
            style={{
              width: 100 - win + '%',
              background: '#0891b2',
              paddingRight: 10,
              textAlign: 'right',
              whiteSpace: 'nowrap',
              transition: 'width 0.2s ease',
            }}
          >
            Defense {100 - win}%
          </div>
        </div>

        {/* Radar Canvas */}
        <canvas
          ref={canvas}
          width={SIZE}
          height={SIZE}
          onMouseMove={onMove}
          style={{
            width: 'min(800px, 60vw)',
            height: 'min(800px, 60vw)',
            borderRadius: 8,
            border: '1px solid #1e293b',
            background: '#020617',
            cursor: 'crosshair',
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          }}
        />
      </div>

      {/* Right Tactical Sidebar */}
      <div style={{ width: 380, background: '#1e293b', padding: 20, borderRadius: 8, height: 'fit-content' }}>
        <h3 style={{ margin: '0 0 12px 0', borderBottom: '1px solid #334155', paddingBottom: 8 }}>
          Round Clock: {frame ? Math.floor(frame.clock) + ' s' : '-'}{' '}
          {frame?.bomb_planted ? <span style={{ color: '#ef4444' }}>(BOMB PLANTED)</span> : ''}
        </h3>

        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase' }}>
            Living Operators & Danger Rating
          </div>
          {alive.map((p) => (
            <div key={p.slot} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span style={{ width: 75, fontWeight: 500, color: p.team === 0 ? '#fb923c' : '#38bdf8' }}>
                P{p.slot + 1} ({p.team === 0 ? 'T' : 'CT'})
              </span>
              <div style={{ flex: 1, height: 8, background: '#334155', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ width: p.hp * 100 + '%', height: 8, background: '#22c55e', borderRadius: 4 }} />
              </div>
              <span style={{ width: 44, textAlign: 'right', fontWeight: 600, color: threatColor(p.threat) }}>
                {Math.round(p.threat * 100)}%
              </span>
            </div>
          ))}
        </div>

        <div style={{ background: '#0f172a', padding: 12, borderRadius: 6, marginBottom: 12 }}>
          <h4 style={{ margin: '0 0 6px 0', color: '#f59e0b', fontSize: 14 }}>Tactical Threat Intel (Hover)</h4>
          <p style={{ margin: 0, fontSize: 13, lineHeight: '1.4', color: hovered ? '#f8fafc' : '#64748b' }}>
            {hovered ? hovered.why || 'No specific threat detected.' : 'Hover over any dot on the radar.'}
          </p>
        </div>

        <div style={{ background: '#0f172a', padding: 12, borderRadius: 6 }}>
          <h4 style={{ margin: '0 0 6px 0', color: '#38bdf8', fontSize: 14 }}>Round Momentum Intel</h4>
          <p style={{ margin: 0, fontSize: 13, lineHeight: '1.4', color: frame?.why_win ? '#f8fafc' : '#64748b' }}>
            {frame?.why_win || 'Awaiting round replay stream...'}
          </p>
        </div>
      </div>
    </div>
  );
}
