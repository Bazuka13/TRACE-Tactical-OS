import { useEffect, useRef, useState, type MouseEvent } from 'react';

type Player = {
  slot: number;
  team: number;
  x: number;
  y: number;
  yaw: number;
  hp: number;
  armor: number;
  weapon_dps: number;
  speed: number;
  alive: boolean;
  threat: number;
  los_bits: number;
  is_isolated: number;
  has_crossfire: number;
  why: string;
};

type Frame = {
  type: string;
  frame_idx: number;
  total_frames: number;
  tick: number;
  map: string;
  clock: number;
  players: Player[];
  territory: number[][];
  danger: number[][];
  win_prob: number;
  why_win: string;
  bomb_planted: number;
  att_power: number;
  def_power: number;
  att_alive: number;
  def_alive: number;
  att_territory: number;
};

type Wall = [[number, number], [number, number]];

const SIZE = 800;
const WS_BASE = 'ws://localhost:8000/ws/replay/';
const API_BASE = 'http://localhost:8000';

const threatColor = (t: number) => {
  if (t >= 0.6) return '#ef4444';
  if (t >= 0.3) return '#f59e0b';
  return '#10b981';
};

const threatBadge = (t: number) => {
  if (t >= 0.6) return { text: 'HIGH', bg: 'rgba(239, 68, 68, 0.2)', border: '#ef4444', color: '#fca5a5' };
  if (t >= 0.3) return { text: 'MED', bg: 'rgba(245, 158, 11, 0.2)', border: '#f59e0b', color: '#fde68a' };
  return { text: 'LOW', bg: 'rgba(16, 185, 129, 0.2)', border: '#10b981', color: '#a7f3d0' };
};

export default function App() {
  const [matches, setMatches] = useState<string[]>([]);
  const [matchId, setMatchId] = useState<string>('Match01_morningstar-vs-fengda-m1-mirage');
  const [rounds, setRounds] = useState<number[]>([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const [round, setRound] = useState<number>(7);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<number | null>(null);
  const [hoverSlot, setHoverSlot] = useState<number | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [walls, setWalls] = useState<Wall[]>([]);

  // Layer Visibility Toggles
  const [showWalls, setShowWalls] = useState(true);
  const [showSightlines, setShowSightlines] = useState(true);
  const [showDanger, setShowDanger] = useState(true);
  const [showTerritory, setShowTerritory] = useState(true);
  const [showCones, setShowCones] = useState(true);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latestFrame = useRef<Frame | null>(null);
  const hoverRef = useRef<number | null>(null);
  const selectedRef = useRef<number | null>(null);
  const mapImg = useRef<HTMLImageElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const wallsRef = useRef<Wall[]>([]);

  // Sync refs for requestAnimationFrame loop
  hoverRef.current = hoverSlot;
  selectedRef.current = selectedSlot;
  wallsRef.current = walls;

  // 1. Fetch available matches from backend
  useEffect(() => {
    fetch(`${API_BASE}/matches`)
      .then((r) => r.json())
      .then((data: string[]) => {
        if (data && data.length > 0) {
          setMatches(data);
          if (!data.includes(matchId)) {
            setMatchId(data[0]);
          }
        }
      })
      .catch((err) => console.warn('Could not load matches list:', err));
  }, []);

  // 2. Fetch rounds for selected match
  useEffect(() => {
    if (!matchId) return;
    fetch(`${API_BASE}/matches/${matchId}/rounds`)
      .then((r) => r.json())
      .then((data: { rounds: number[] }) => {
        if (data.rounds && data.rounds.length > 0) {
          setRounds(data.rounds);
          if (!data.rounds.includes(round)) {
            setRound(data.rounds[0]);
          }
        }
      })
      .catch((err) => console.warn('Could not load rounds:', err));
  }, [matchId]);

  // 3. Canvas Render Loop
  useEffect(() => {
    let animId = 0;
    const render = () => {
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) {
        drawCanvas(
          ctx,
          latestFrame.current,
          mapImg.current,
          hoverRef.current,
          selectedRef.current,
          wallsRef.current,
          { showWalls, showSightlines, showDanger, showTerritory, showCones }
        );
      }
      animId = requestAnimationFrame(render);
    };
    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [showWalls, showSightlines, showDanger, showTerritory, showCones]);

  // 4. WebSocket Start & Stream handler
  const startStream = () => {
    wsRef.current?.close();
    const url = `${WS_BASE}${matchId}/${round}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    setStatus('connecting');

    ws.onopen = () => setStatus('live');
    ws.onclose = () => setStatus('finished');
    ws.onerror = () => setStatus('error');

    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.type === 'meta') {
        if (d.walls) setWalls(d.walls);
        if (!mapImg.current || mapImg.current.dataset.map !== d.map) {
          const img = new Image();
          img.src = `/maps/${d.map}.png`;
          img.dataset.map = d.map;
          mapImg.current = img;
        }
        return;
      }
      if (d.done) {
        setStatus('round completed');
        return;
      }
      if (d.error) {
        setStatus(`error: ${d.error}`);
        return;
      }
      latestFrame.current = d;
      setFrame(d);
    };
  };

  const stopStream = () => {
    wsRef.current?.close();
    setStatus('paused');
  };

  const handleCanvasMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * SIZE;
    const my = ((e.clientY - rect.top) / rect.height) * SIZE;
    let closest: number | null = null;
    let minD = 25;
    for (const p of latestFrame.current?.players ?? []) {
      if (!p.alive) continue;
      const d = Math.hypot(p.x * SIZE - mx, p.y * SIZE - my);
      if (d < minD) {
        minD = d;
        closest = p.slot;
      }
    }
    setHoverSlot(closest);
  };

  const handleCanvasClick = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * SIZE;
    const my = ((e.clientY - rect.top) / rect.height) * SIZE;
    let clicked: number | null = null;
    let minD = 28;
    for (const p of latestFrame.current?.players ?? []) {
      if (!p.alive) continue;
      const d = Math.hypot(p.x * SIZE - mx, p.y * SIZE - my);
      if (d < minD) {
        minD = d;
        clicked = p.slot;
      }
    }
    setSelectedSlot(clicked);
  };

  // Metrics computation
  const activeSlot = selectedSlot !== null ? selectedSlot : hoverSlot;
  const inspectedPlayer = frame?.players.find((p) => p.slot === activeSlot);
  const winPct = frame ? Math.round(frame.win_prob * 100) : 50;
  const alivePlayers = (frame?.players ?? []).filter((p) => p.alive).sort((a, b) => b.threat - a.threat);
  const highThreatCount = alivePlayers.filter((p) => p.threat >= 0.6).length;
  const progressPct = frame && frame.total_frames > 0 ? (frame.frame_idx / frame.total_frames) * 100 : 0;

  return (
    <div style={{ padding: '16px 24px', maxWidth: 1720, margin: '0 auto' }}>
      {/* Top Header */}
      <header
        className="glass-panel"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 24px',
          marginBottom: 18,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 8,
              background: 'linear-gradient(135deg, #f97316, #06b6d4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 800,
              fontSize: 18,
              boxShadow: '0 0 16px rgba(249, 115, 22, 0.4)',
            }}
          >
            T
          </div>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: '0.04em', margin: 0 }}>
              ESPORTS TACTICAL OS <span style={{ fontSize: 12, color: '#38bdf8', fontWeight: 500 }}>v1.0-LIVE</span>
            </h1>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>
              CS2 Replay Analytics, Threat Forecasting & Neural SHAP Engine
            </span>
          </div>
        </div>

        {/* Replay Controls & Match Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8' }}>MATCH:</span>
            <select
              value={matchId}
              onChange={(e) => setMatchId(e.target.value)}
              style={{
                background: '#1e293b',
                color: '#f8fafc',
                border: '1px solid #475569',
                borderRadius: 6,
                padding: '6px 10px',
                fontSize: 12,
                fontFamily: "'Outfit', sans-serif",
                cursor: 'pointer',
              }}
            >
              {matches.map((m) => (
                <option key={m} value={m}>
                  {m.length > 34 ? m.slice(0, 32) + '...' : m}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8' }}>ROUND:</span>
            <select
              value={round}
              onChange={(e) => setRound(Number(e.target.value))}
              style={{
                background: '#1e293b',
                color: '#f8fafc',
                border: '1px solid #475569',
                borderRadius: 6,
                padding: '6px 10px',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              {rounds.map((r) => (
                <option key={r} value={r}>
                  Round {r}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={status === 'live' ? stopStream : startStream}
            style={{
              background: status === 'live' ? '#dc2626' : '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              padding: '7px 18px',
              fontWeight: 600,
              fontSize: 13,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              boxShadow: status === 'live' ? '0 0 12px rgba(220,38,38,0.5)' : '0 0 12px rgba(37,99,235,0.4)',
              transition: 'all 0.2s',
            }}
          >
            {status === 'live' ? '■ Pause' : '▶ Play Replay'}
          </button>

          <div
            style={{
              padding: '4px 10px',
              borderRadius: 20,
              fontSize: 11,
              fontWeight: 700,
              background:
                status === 'live'
                  ? 'rgba(16, 185, 129, 0.2)'
                  : status === 'error'
                  ? 'rgba(239, 68, 68, 0.2)'
                  : 'rgba(100, 116, 139, 0.2)',
              color: status === 'live' ? '#10b981' : status === 'error' ? '#ef4444' : '#94a3b8',
              border: `1px solid ${status === 'live' ? '#10b981' : '#475569'}`,
            }}
          >
            {status.toUpperCase()}
          </div>
        </div>
      </header>

      {/* Main Grid: 2 Columns */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 840px) minmax(400px, 1fr)', gap: 20 }}>
        {/* Left Column: Replay Viewport */}
        <div>
          {/* Win Probability Duel Meter */}
          <div className="glass-panel" style={{ padding: '12px 16px', marginBottom: 14 }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: 12,
                fontWeight: 700,
                marginBottom: 6,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f97316' }}>
                <span>ATTACK (T)</span>
                <span className="font-mono" style={{ fontSize: 16 }}>{winPct}%</span>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>
                  Alive: {frame?.att_alive ?? 5} | Power: {frame?.att_power?.toFixed(2) ?? '-'}
                </span>
              </div>

              {frame?.bomb_planted === 1 && (
                <div
                  className="pulse-alert font-mono"
                  style={{
                    background: '#dc2626',
                    color: '#fff',
                    padding: '2px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: 700,
                  }}
                >
                  BOMB PLANTED
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#06b6d4' }}>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>
                  Power: {frame?.def_power?.toFixed(2) ?? '-'} | Alive: {frame?.def_alive ?? 5}
                </span>
                <span className="font-mono" style={{ fontSize: 16 }}>{100 - winPct}%</span>
                <span>DEFENSE (CT)</span>
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                height: 10,
                borderRadius: 5,
                overflow: 'hidden',
                background: '#1e293b',
              }}
            >
              <div
                style={{
                  width: `${winPct}%`,
                  background: 'linear-gradient(90deg, #ea580c, #f97316)',
                  transition: 'width 0.15s ease-out',
                }}
              />
              <div
                style={{
                  width: `${100 - winPct}%`,
                  background: 'linear-gradient(90deg, #06b6d4, #0891b2)',
                  transition: 'width 0.15s ease-out',
                }}
              />
            </div>
          </div>

          {/* Radar Canvas with interactive overlays */}
          <div className="glass-panel" style={{ padding: 14, position: 'relative' }}>
            {/* Layer Toggles Toolbar */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                marginBottom: 10,
                flexWrap: 'wrap',
              }}
            >
              <span style={{ fontSize: 11, fontWeight: 700, color: '#64748b', marginRight: 4 }}>
                OVERLAYS:
              </span>
              <div
                className={`toggle-chip ${showWalls ? 'active' : ''}`}
                onClick={() => setShowWalls(!showWalls)}
              >
                Obstacle Walls
              </div>
              <div
                className={`toggle-chip ${showSightlines ? 'active' : ''}`}
                onClick={() => setShowSightlines(!showSightlines)}
              >
                Sightlines (LOS)
              </div>
              <div
                className={`toggle-chip ${showDanger ? 'active' : ''}`}
                onClick={() => setShowDanger(!showDanger)}
              >
                Danger Heatmap
              </div>
              <div
                className={`toggle-chip ${showTerritory ? 'active' : ''}`}
                onClick={() => setShowTerritory(!showTerritory)}
              >
                Territory Control
              </div>
              <div
                className={`toggle-chip ${showCones ? 'active' : ''}`}
                onClick={() => setShowCones(!showCones)}
              >
                FOV Cones
              </div>
            </div>

            {/* Radar Viewport */}
            <div style={{ position: 'relative', width: '100%', display: 'flex', justifyContent: 'center' }}>
              <canvas
                ref={canvasRef}
                width={SIZE}
                height={SIZE}
                onMouseMove={handleCanvasMouseMove}
                onClick={handleCanvasClick}
                style={{
                  width: '100%',
                  maxWidth: 800,
                  aspectRatio: '1/1',
                  borderRadius: 10,
                  border: '1px solid #334155',
                  background: '#040711',
                  cursor: 'crosshair',
                  boxShadow: 'inset 0 0 40px rgba(0,0,0,0.8)',
                }}
              />
            </div>

            {/* Replay Timeline Scrubber */}
            <div style={{ marginTop: 12 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                }}
                className="font-mono"
              >
                <span>
                  ROUND CLOCK: <strong style={{ color: '#f8fafc' }}>{frame ? `${frame.clock.toFixed(1)}s` : '0.0s'}</strong>
                </span>
                <span>
                  FRAME: {frame ? frame.frame_idx : 0} / {frame ? frame.total_frames : 0}
                </span>
                <span>
                  TICK: <strong style={{ color: '#38bdf8' }}>{frame ? frame.tick : '-'}</strong>
                </span>
              </div>
              <div
                style={{
                  height: 6,
                  borderRadius: 3,
                  background: '#1e293b',
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    width: `${progressPct}%`,
                    height: '100%',
                    background: '#38bdf8',
                    transition: 'width 0.1s linear',
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Tactical Intelligence, Telemetry & Live Stats */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* AI Coach Tactical Threat Intelligence Card */}
          <div className="glass-panel" style={{ padding: 18 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid var(--border-color)',
                paddingBottom: 10,
                marginBottom: 12,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 16 }}>🎯</span>
                <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0, letterSpacing: '0.03em' }}>
                  AI THREAT RADAR INTEL
                </h2>
              </div>
              {inspectedPlayer && (
                <div
                  style={{
                    padding: '3px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: 700,
                    background: threatBadge(inspectedPlayer.threat).bg,
                    border: `1px solid ${threatBadge(inspectedPlayer.threat).border}`,
                    color: threatBadge(inspectedPlayer.threat).color,
                  }}
                >
                  {threatBadge(inspectedPlayer.threat).text} DANGER ({Math.round(inspectedPlayer.threat * 100)}%)
                </div>
              )}
            </div>

            {inspectedPlayer ? (
              <div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    marginBottom: 12,
                  }}
                >
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 6,
                      background: inspectedPlayer.team === 0 ? '#f97316' : '#06b6d4',
                      color: '#000',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontWeight: 800,
                      fontSize: 14,
                    }}
                  >
                    P{inspectedPlayer.slot + 1}
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>
                      Operator #{inspectedPlayer.slot + 1} ({inspectedPlayer.team === 0 ? 'Terrorist' : 'Counter-Terrorist'})
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8' }} className="font-mono">
                      POS: ({inspectedPlayer.x.toFixed(3)}, {inspectedPlayer.y.toFixed(3)}) | YAW: {inspectedPlayer.yaw}°
                    </div>
                  </div>
                </div>

                {/* Natural Language Coach Insight */}
                <div
                  style={{
                    background: 'rgba(2, 6, 23, 0.7)',
                    border: '1px solid rgba(56, 189, 248, 0.2)',
                    borderRadius: 8,
                    padding: 12,
                    fontSize: 13,
                    lineHeight: '1.5',
                    color: '#e2e8f0',
                  }}
                >
                  <strong style={{ color: '#38bdf8', display: 'block', marginBottom: 4, fontSize: 11 }}>
                    COACH DIAGNOSTIC:
                  </strong>
                  {inspectedPlayer.why || 'Operator in safe corridor with active cover.'}
                </div>
              </div>
            ) : (
              <div style={{ padding: 20, textAlign: 'center', color: '#64748b', fontSize: 13 }}>
                Hover or click any operator on the radar or roster table to inspect live threat factors.
              </div>
            )}
          </div>

          {/* Round Momentum & Win Predictor Factor Breakdown */}
          <div className="glass-panel" style={{ padding: 18 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                borderBottom: '1px solid var(--border-color)',
                paddingBottom: 10,
                marginBottom: 12,
              }}
            >
              <span style={{ fontSize: 16 }}>⚡</span>
              <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0, letterSpacing: '0.03em' }}>
                ROUND MOMENTUM REASONING
              </h2>
            </div>
            <div
              style={{
                background: 'rgba(2, 6, 23, 0.7)',
                borderRadius: 8,
                padding: 12,
                fontSize: 13,
                lineHeight: '1.5',
                color: '#cbd5e1',
                border: '1px solid rgba(51, 65, 85, 0.4)',
              }}
            >
              {frame?.why_win || 'Model analyzing round dynamics...'}
            </div>

            {/* Quick Metrics Grid */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: 10,
                marginTop: 14,
              }}
            >
              <div
                style={{
                  background: 'rgba(15, 23, 42, 0.6)',
                  padding: 10,
                  borderRadius: 6,
                  textAlign: 'center',
                }}
              >
                <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>TERRITORY</span>
                <span className="font-mono" style={{ fontSize: 15, fontWeight: 700, color: '#f8fafc' }}>
                  {frame ? `${Math.round(frame.att_territory * 100)}% T` : '-'}
                </span>
              </div>
              <div
                style={{
                  background: 'rgba(15, 23, 42, 0.6)',
                  padding: 10,
                  borderRadius: 6,
                  textAlign: 'center',
                }}
              >
                <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>HIGH THREATS</span>
                <span
                  className="font-mono"
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    color: highThreatCount > 0 ? '#ef4444' : '#10b981',
                  }}
                >
                  {highThreatCount} OP
                </span>
              </div>
              <div
                style={{
                  background: 'rgba(15, 23, 42, 0.6)',
                  padding: 10,
                  borderRadius: 6,
                  textAlign: 'center',
                }}
              >
                <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>MAN ADVANTAGE</span>
                <span className="font-mono" style={{ fontSize: 15, fontWeight: 700, color: '#38bdf8' }}>
                  {frame ? `${frame.att_alive}v${frame.def_alive}` : '-'}
                </span>
              </div>
            </div>
          </div>

          {/* Living Operators Roster & Telemetry Table */}
          <div className="glass-panel" style={{ padding: 18, flex: 1 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid var(--border-color)',
                paddingBottom: 10,
                marginBottom: 12,
              }}
            >
              <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0 }}>
                OPERATOR TELEMETRY ROSTER ({alivePlayers.length} ALIVE)
              </h2>
              <span style={{ fontSize: 11, color: '#64748b' }}>Click row to pin</span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: '#64748b', textAlign: 'left', borderBottom: '1px solid #334155' }}>
                    <th style={{ padding: '6px 8px' }}>OP</th>
                    <th style={{ padding: '6px 8px' }}>HP</th>
                    <th style={{ padding: '6px 8px' }}>SPD</th>
                    <th style={{ padding: '6px 8px' }}>COVER</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>DANGER</th>
                  </tr>
                </thead>
                <tbody>
                  {(frame?.players ?? []).map((p) => {
                    const isSelected = selectedSlot === p.slot;
                    const isHovered = hoverSlot === p.slot;
                    return (
                      <tr
                        key={p.slot}
                        onClick={() => setSelectedSlot(p.slot === selectedSlot ? null : p.slot)}
                        onMouseEnter={() => setHoverSlot(p.slot)}
                        onMouseLeave={() => setHoverSlot(null)}
                        style={{
                          cursor: 'pointer',
                          background: isSelected
                            ? 'rgba(56, 189, 248, 0.15)'
                            : isHovered
                            ? 'rgba(255, 255, 255, 0.05)'
                            : 'transparent',
                          opacity: p.alive ? 1 : 0.35,
                          borderBottom: '1px solid rgba(51, 65, 85, 0.3)',
                          transition: 'background 0.15s',
                        }}
                      >
                        <td style={{ padding: '7px 8px', fontWeight: 600 }}>
                          <span
                            style={{
                              color: p.team === 0 ? '#f97316' : '#06b6d4',
                              marginRight: 6,
                            }}
                          >
                            ●
                          </span>
                          P{p.slot + 1}
                        </td>
                        <td style={{ padding: '7px 8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div
                              style={{
                                width: 44,
                                height: 5,
                                background: '#334155',
                                borderRadius: 3,
                                overflow: 'hidden',
                              }}
                            >
                              <div
                                style={{
                                  width: `${p.hp * 100}%`,
                                  height: '100%',
                                  background: p.hp > 0.5 ? '#10b981' : p.hp > 0.25 ? '#f59e0b' : '#ef4444',
                                }}
                              />
                            </div>
                            <span className="font-mono" style={{ fontSize: 11 }}>
                              {Math.round(p.hp * 100)}
                            </span>
                          </div>
                        </td>
                        <td style={{ padding: '7px 8px' }} className="font-mono">
                          {p.speed > 0 ? `${Math.round(p.speed)}` : '0'}
                        </td>
                        <td style={{ padding: '7px 8px' }}>
                          {p.is_isolated === 1 ? (
                            <span style={{ color: '#ef4444', fontSize: 10, fontWeight: 600 }}>ISOLATED</span>
                          ) : p.has_crossfire === 1 ? (
                            <span style={{ color: '#10b981', fontSize: 10, fontWeight: 600 }}>CROSSFIRE</span>
                          ) : (
                            <span style={{ color: '#94a3b8', fontSize: 10 }}>SUPPORT</span>
                          )}
                        </td>
                        <td style={{ padding: '7px 8px', textAlign: 'right' }}>
                          <span
                            className="font-mono"
                            style={{
                              fontWeight: 700,
                              color: threatColor(p.threat),
                            }}
                          >
                            {Math.round(p.threat * 100)}%
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Canvas Drawing Engine
function drawCanvas(
  ctx: CanvasRenderingContext2D,
  f: Frame | null,
  img: HTMLImageElement | null,
  hover: number | null,
  selected: number | null,
  walls: Wall[],
  opts: {
    showWalls: boolean;
    showSightlines: boolean;
    showDanger: boolean;
    showTerritory: boolean;
    showCones: boolean;
  }
) {
  ctx.clearRect(0, 0, SIZE, SIZE);

  // 1. Radar Base Map
  if (img && img.complete) {
    ctx.drawImage(img, 0, 0, SIZE, SIZE);
  } else {
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, SIZE, SIZE);
  }

  if (!f) return;

  // 2. Voronoi Territory Control Overlay
  if (opts.showTerritory && f.territory && f.territory.length > 0) {
    const step = SIZE / f.territory.length;
    f.territory.forEach((row, r) => {
      row.forEach((team, c) => {
        if (team < 0) return;
        ctx.fillStyle = team === 0 ? 'rgba(249, 115, 22, 0.16)' : 'rgba(6, 182, 212, 0.16)';
        ctx.fillRect(c * step, r * step, step + 0.5, step + 0.5);
      });
    });
  }

  // 3. Danger Heatmap Overlay
  if (opts.showDanger && f.danger && f.danger.length > 0) {
    const dstep = SIZE / f.danger.length;
    f.danger.forEach((row, r) => {
      row.forEach((val, c) => {
        if (val < 0.25) return;
        ctx.fillStyle = `rgba(239, 68, 68, ${val * 0.45})`;
        ctx.fillRect(c * dstep, r * dstep, dstep + 0.5, dstep + 0.5);
      });
    });
  }

  // 4. Yellow Obstacle Walls
  if (opts.showWalls && walls && walls.length > 0) {
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#eab308'; // yellow
    for (const [p1, p2] of walls) {
      ctx.beginPath();
      ctx.moveTo(p1[0] * SIZE, p1[1] * SIZE);
      ctx.lineTo(p2[0] * SIZE, p2[1] * SIZE);
      ctx.stroke();
    }
  }

  // 5. Active Line-of-Sight Rays
  if (opts.showSightlines) {
    const players = f.players;
    for (let i = 0; i < players.length; i++) {
      const p1 = players[i];
      if (!p1.alive) continue;
      for (let j = i + 1; j < players.length; j++) {
        const p2 = players[j];
        if (!p2.alive) continue;
        const sees = (p1.los_bits >> p2.slot) & 1;
        if (sees) {
          ctx.beginPath();
          ctx.moveTo(p1.x * SIZE, p1.y * SIZE);
          ctx.lineTo(p2.x * SIZE, p2.y * SIZE);
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = p1.team === p2.team ? 'rgba(34, 197, 94, 0.65)' : 'rgba(239, 68, 68, 0.75)';
          ctx.stroke();
        }
      }
    }
  }

  // 6. Players, Directional Cones & Pulsing Danger Rings
  const now = performance.now();
  for (const p of f.players) {
    if (!p.alive) continue;
    const px = p.x * SIZE;
    const py = p.y * SIZE;
    const rad = (p.yaw * Math.PI) / 180;
    const isTargeted = hover === p.slot || selected === p.slot;

    // FOV Cone
    if (opts.showCones) {
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.arc(px, py, 55, rad - Math.PI / 4, rad + Math.PI / 4);
      ctx.closePath();
      ctx.fillStyle = p.team === 0 ? 'rgba(249, 115, 22, 0.12)' : 'rgba(6, 182, 212, 0.12)';
      ctx.fill();
    }

    // High Danger Pulse Ring (when threat >= 0.6)
    if (p.threat >= 0.6) {
      const pulse = 4 * Math.sin(now / 120);
      ctx.beginPath();
      ctx.arc(px, py, 16 + pulse, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
      ctx.lineWidth = 3;
      ctx.stroke();
    } else if (p.threat >= 0.3) {
      ctx.beginPath();
      ctx.arc(px, py, 15, 0, 2 * Math.PI);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Selected / Hovered crosshair ring
    if (isTargeted) {
      ctx.beginPath();
      ctx.arc(px, py, 22, 0, 2 * Math.PI);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Operator Core Dot
    ctx.beginPath();
    ctx.arc(px, py, 9, 0, 2 * Math.PI);
    ctx.globalAlpha = 0.4 + 0.6 * p.hp;
    ctx.fillStyle = p.team === 0 ? '#f97316' : '#06b6d4';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000';
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Player Number Tag
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${p.slot + 1}`, px, py);
  }
}
