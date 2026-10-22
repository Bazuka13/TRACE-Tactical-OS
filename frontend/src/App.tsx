import { useEffect, useRef, useState, type MouseEvent } from 'react';

type Factor = {
  text: string;
  up: boolean;
  share: number;
};

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
  dist_enemy?: number;
  los_bits: number;
  is_isolated: number;
  has_crossfire: number;
  in_enemy_zone: number;
  why: string;
  factors?: Factor[];
};

type Smoke = {
  x: number;
  y: number;
  r: number;
  progress: number;
};

type Death = {
  slot: number;
  team: number;
  tick: number;
  x: number;
  y: number;
  killer: number;
};

type ExtraAnalytics = {
  economy_tier_t: string;
  economy_tier_ct: string;
  risk_t: number;
  risk_ct: number;
  predicted_site: string;
  site_prob_a: number;
  retake_odds: number;
  clutch: { clutcher: string; situation: string; odds: number } | null;
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
  win_factors: Factor[];
  bomb_planted: number;
  att_power: number;
  def_power: number;
  att_alive: number;
  def_alive: number;
  att_territory: number;
  smokes: Smoke[];
  deaths: Death[];
  recent_kills: Death[];
  extra: ExtraAnalytics;
};

type Wall = [[number, number], [number, number]];

type TacticalLogItem = {
  id: number;
  time: string;
  type: 'kill' | 'threat' | 'bomb' | 'smoke';
  text: string;
  color: string;
};

const SIZE = 800;
const envWs = import.meta.env.VITE_WS_URL;
const WS_BASE = envWs
  ? envWs.startsWith('/')
    ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${envWs}`
    : envWs
  : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8000/ws/replay/`;

const envApi = import.meta.env.VITE_API_URL;
const API_BASE = envApi !== undefined && envApi !== ''
  ? envApi
  : `${window.location.protocol === 'https:' ? 'https:' : 'http:'}//${window.location.hostname}:8000`;

const threatColor = (t: number) => {
  if (t >= 0.6) return '#ef4444';
  if (t >= 0.3) return '#f59e0b';
  return '#10b981';
};

const threatBadge = (t: number) => {
  if (t >= 0.6) return { text: 'CRITICAL', bg: 'rgba(239, 68, 68, 0.25)', border: '#ef4444', color: '#fca5a5' };
  if (t >= 0.3) return { text: 'ELEVATED', bg: 'rgba(245, 158, 11, 0.25)', border: '#f59e0b', color: '#fde68a' };
  return { text: 'LOW RISK', bg: 'rgba(16, 185, 129, 0.25)', border: '#10b981', color: '#a7f3d0' };
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
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1);
  const [walls, setWalls] = useState<Wall[]>([]);
  const [tacticalLogs, setTacticalLogs] = useState<TacticalLogItem[]>([]);

  // Paced / Latched AI Intelligence State (ensures easy reading)
  const [latchedThreatText, setLatchedThreatText] = useState<string>('');
  const [latchedThreatFactors, setLatchedThreatFactors] = useState<Factor[]>([]);
  const [latchedWinText, setLatchedWinText] = useState<string>('');
  const [latchedWinFactors, setLatchedWinFactors] = useState<Factor[]>([]);
  const lastIntelUpdateRef = useRef<number>(0);

  // Layer Toggles
  const [showWalls, setShowWalls] = useState(true);
  const [showSightlines, setShowSightlines] = useState(true);
  const [showDanger, setShowDanger] = useState(true);
  const [showTerritory, setShowTerritory] = useState(false);
  const [showCones, setShowCones] = useState(true);
  const [showSmokes, setShowSmokes] = useState(true);
  const [showDeaths, setShowDeaths] = useState(true);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latestFrame = useRef<Frame | null>(null);
  const hoverRef = useRef<number | null>(null);
  const selectedRef = useRef<number | null>(null);
  const mapImg = useRef<HTMLImageElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const wallsRef = useRef<Wall[]>([]);
  const killAnimationsRef = useRef<{ x: number; y: number; startTime: number }[]>([]);
  const knownDeathCountRef = useRef<number>(0);

  hoverRef.current = hoverSlot;
  selectedRef.current = selectedSlot;
  wallsRef.current = walls;

  // Preload map image on mount
  useEffect(() => {
    const img = new Image();
    img.src = '/maps/de_mirage.png';
    img.dataset.map = 'de_mirage';
    mapImg.current = img;
  }, []);

  // 1. Fetch available matches
  useEffect(() => {
    fetch(`${API_BASE}/matches`)
      .then((r) => r.json())
      .then((data: string[]) => {
        if (data && data.length > 0) {
          setMatches(data);
          if (!data.includes(matchId)) setMatchId(data[0]);
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
          if (!data.rounds.includes(round)) setRound(data.rounds[0]);
        }
      })
      .catch((err) => console.warn('Could not load rounds:', err));
  }, [matchId]);

  // 3. Load initial frame preview whenever match or round changes
  useEffect(() => {
    if (!matchId || !round) return;
    fetch(`${API_BASE}/matches/${matchId}/${round}/initial_frame`)
      .then((r) => r.json())
      .then((data) => {
        if (data.meta) {
          if (data.meta.walls) setWalls(data.meta.walls);
          if (!mapImg.current || mapImg.current.dataset.map !== data.meta.map) {
            const img = new Image();
            img.src = `/maps/${data.meta.map}.png`;
            img.dataset.map = data.meta.map;
            mapImg.current = img;
          }
        }
        if (data.frame) {
          setFrame(data.frame);
          latestFrame.current = data.frame;
          if (data.frame.why_win) setLatchedWinText(data.frame.why_win);
          if (data.frame.win_factors) setLatchedWinFactors(data.frame.win_factors);
          if (data.frame.players && data.frame.players[0]) {
            setLatchedThreatText(data.frame.players[0].why);
            setLatchedThreatFactors(data.frame.players[0].factors || []);
            setSelectedSlot((prev) => (prev === null ? data.frame.players[0].slot : prev));
          }
        }
      })
      .catch((err) => console.warn('Could not load initial frame:', err));
  }, [matchId, round]);

  // 4. Canvas Render Loop with 60 FPS interpolation & animations
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
          killAnimationsRef.current,
          {
            showWalls,
            showSightlines,
            showDanger,
            showTerritory,
            showCones,
            showSmokes,
            showDeaths,
          }
        );
      }
      animId = requestAnimationFrame(render);
    };
    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [showWalls, showSightlines, showDanger, showTerritory, showCones, showSmokes, showDeaths]);

  // 5. Speed and Seek Controllers
  const changeSpeed = (s: number) => {
    setSpeedMultiplier(s);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'set_speed', speed: s }));
    }
  };

  const handleSeek = (targetFrame: number) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'seek', frame: targetFrame }));
    } else {
      startStream();
      setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'seek', frame: targetFrame }));
        }
      }, 300);
    }
  };

  // 6. WebSocket Controller
  const startStream = () => {
    wsRef.current?.close();
    setTacticalLogs([]);
    killAnimationsRef.current = [];
    knownDeathCountRef.current = 0;

    const url = `${WS_BASE}${matchId}/${round}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    setStatus('connecting');

    ws.onopen = () => {
      setStatus('live');
      if (speedMultiplier !== 1) {
        ws.send(JSON.stringify({ type: 'set_speed', speed: speedMultiplier }));
      }
    };
    ws.onclose = () => setStatus('finished');
    ws.onerror = () => setStatus('error');

    ws.onmessage = (e) => {
      const d: Frame = JSON.parse(e.data);
      if (d.type === 'meta') {
        const meta = d as any;
        if (meta.walls) setWalls(meta.walls);
        if (!mapImg.current || mapImg.current.dataset.map !== meta.map) {
          const img = new Image();
          img.src = `/maps/${meta.map}.png`;
          img.dataset.map = meta.map;
          mapImg.current = img;
        }
        return;
      }
      if ((d as any).done) {
        setStatus('round completed');
        return;
      }
      if ((d as any).error) {
        setStatus(`error: ${(d as any).error}`);
        return;
      }

      // Detect new deaths for ripple animations and tactical event log
      if (d.deaths && d.deaths.length > knownDeathCountRef.current) {
        const newDeaths = d.deaths.slice(knownDeathCountRef.current);
        knownDeathCountRef.current = d.deaths.length;
        const now = performance.now();
        newDeaths.forEach((k) => {
          killAnimationsRef.current.push({ x: k.x * SIZE, y: k.y * SIZE, startTime: now });
          const killerLabel = k.killer >= 0 ? `P${k.killer + 1}` : 'Unknown';
          const victimLabel = `P${k.slot + 1} (${k.team === 0 ? 'T' : 'CT'})`;
          setTacticalLogs((prev) => [
            {
              id: Date.now() + Math.random(),
              time: `${d.clock.toFixed(1)}s`,
              type: 'kill',
              text: `${killerLabel} ELIMINATED ${victimLabel}`,
              color: '#ef4444',
            },
            ...prev.slice(0, 19),
          ]);
        });
      }

      // Check for Bomb Plant
      if (d.bomb_planted === 1 && latestFrame.current?.bomb_planted === 0) {
        setTacticalLogs((prev) => [
          {
            id: Date.now() + Math.random(),
            time: `${d.clock.toFixed(1)}s`,
            type: 'bomb',
            text: 'C4 EXPLOSIVE PLANTED - DEFENDERS MUST RETAKE',
            color: '#f97316',
          },
          ...prev.slice(0, 19),
        ]);
      }

      // Smooth / Paced SHAP Intelligence Updater (updates at most every 2.5s or on player switch)
      const now = performance.now();
      const currentActive = selectedRef.current !== null ? selectedRef.current : hoverRef.current;
      const targetPlayer = currentActive !== null ? d.players.find((p) => p.slot === currentActive) : null;
      const topThreatPlayer = d.players.filter((p) => p.alive).sort((a, b) => b.threat - a.threat)[0];
      const activeInspected = targetPlayer || topThreatPlayer;

      if (now - lastIntelUpdateRef.current > 2500 || (targetPlayer && targetPlayer.slot !== currentActive)) {
        if (activeInspected) {
          setLatchedThreatText(activeInspected.why || 'Operator in standard corridor formation.');
          setLatchedThreatFactors(activeInspected.factors || []);
        }
        if (d.why_win) {
          setLatchedWinText(d.why_win);
          setLatchedWinFactors(d.win_factors || []);
        }
        lastIntelUpdateRef.current = now;
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
    let minD = 28;
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
    let minD = 32;
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

  // Active inspected operator
  const activeSlot = selectedSlot !== null ? selectedSlot : hoverSlot;
  const inspectedPlayer = frame?.players.find((p) => p.slot === activeSlot) ||
    (frame?.players.filter((p) => p.alive).sort((a, b) => b.threat - a.threat)[0] ?? null);

  const winPct = frame ? Math.round(frame.win_prob * 100) : 50;
  const alivePlayers = (frame?.players ?? []).filter((p) => p.alive);
  const progressPct = frame && frame.total_frames > 0 ? (frame.frame_idx / frame.total_frames) * 100 : 0;

  return (
    <div style={{ padding: '16px 28px', maxWidth: 1780, margin: '0 auto' }}>
      {/* Top Professional Esports Broadcast Header */}
      <header
        className="glass-panel"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 24px',
          marginBottom: 16,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 8,
              background: 'linear-gradient(135deg, #f97316 0%, #06b6d4 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 900,
              fontSize: 20,
              color: '#000',
              boxShadow: '0 0 20px rgba(249, 115, 22, 0.5)',
            }}
          >
            ◈
          </div>
          <div>
            <h1 style={{ fontSize: 19, fontWeight: 800, letterSpacing: '0.04em', margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
              ESPORTS TACTICAL OS
              <span style={{ fontSize: 11, background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', padding: '2px 8px', borderRadius: 4, border: '1px solid rgba(56, 189, 248, 0.3)' }}>
                PRO OBSERVATION SUITE
              </span>
            </h1>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              CS2 Neural Replay Analytics, Multi-Target Ambush Forecasting & Explainable AI
            </span>
          </div>
        </div>

        {/* Control Center */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Match selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-dim)' }}>MATCH:</span>
            <select
              value={matchId}
              onChange={(e) => setMatchId(e.target.value)}
              style={{
                background: '#131b2e',
                color: '#f8fafc',
                border: '1px solid #334155',
                borderRadius: 6,
                padding: '6px 10px',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              {matches.map((m) => (
                <option key={m} value={m}>
                  {m.length > 36 ? m.slice(0, 34) + '...' : m}
                </option>
              ))}
            </select>
          </div>

          {/* Round selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-dim)' }}>ROUND:</span>
            <select
              value={round}
              onChange={(e) => setRound(Number(e.target.value))}
              style={{
                background: '#131b2e',
                color: '#f8fafc',
                border: '1px solid #334155',
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

          {/* Play / Pause button */}
          <button
            onClick={status === 'live' ? stopStream : startStream}
            style={{
              background: status === 'live' ? '#ef4444' : '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              padding: '8px 20px',
              fontWeight: 700,
              fontSize: 13,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              boxShadow: status === 'live' ? '0 0 16px rgba(239, 68, 68, 0.45)' : '0 0 16px rgba(37, 99, 235, 0.45)',
              transition: 'all 0.2s',
            }}
          >
            {status === 'live' ? '■ Pause Replay' : '▶ Play Round Stream'}
          </button>

          {/* Speed chips */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(11, 17, 32, 0.6)', padding: 3, borderRadius: 6, border: '1px solid #334155' }}>
            {[0.5, 1, 2, 4].map((s) => (
              <div
                key={s}
                className={`speed-chip ${speedMultiplier === s ? 'active' : ''}`}
                onClick={() => changeSpeed(s)}
              >
                {s}x
              </div>
            ))}
          </div>

          {/* Status badge */}
          <div
            style={{
              padding: '5px 12px',
              borderRadius: 20,
              fontSize: 11,
              fontWeight: 800,
              background:
                status === 'live'
                  ? 'rgba(16, 185, 129, 0.2)'
                  : status === 'error'
                  ? 'rgba(239, 68, 68, 0.2)'
                  : 'rgba(100, 116, 139, 0.2)',
              color: status === 'live' ? '#10b981' : status === 'error' ? '#ef4444' : '#94a3b8',
              border: `1px solid ${status === 'live' ? '#10b981' : '#475569'}`,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: status === 'live' ? '#10b981' : '#94a3b8' }} />
            {status.toUpperCase()}
          </div>
        </div>
      </header>

      {/* Main Grid: Viewport & Right Analytics Panel */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 840px) minmax(460px, 1fr)', gap: 20 }}>
        {/* Left Column: Replay Viewport & Telemetry Matrix */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Top Live Prediction Banner & Dual Momentum Gauge */}
          <div className="glass-panel" style={{ padding: '14px 18px' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 8,
              }}
            >
              {/* T Team side */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span
                  style={{
                    background: '#f97316',
                    color: '#000',
                    fontWeight: 900,
                    fontSize: 11,
                    padding: '2px 7px',
                    borderRadius: 4,
                  }}
                >
                  TERRORISTS
                </span>
                <span className="font-mono" style={{ fontSize: 20, fontWeight: 800, color: '#f97316' }}>
                  {winPct}%
                </span>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>
                  Alive: <strong style={{ color: '#f8fafc' }}>{frame?.att_alive ?? 5}</strong> | Buy:{' '}
                  <span style={{ color: '#fba768' }}>{frame?.extra?.economy_tier_t ?? 'FULL BUY'}</span>
                </div>
              </div>

              {/* Center status: Bomb plant or Site prediction */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                {frame?.bomb_planted === 1 ? (
                  <div
                    className="pulse-alert font-mono"
                    style={{
                      background: '#ef4444',
                      color: '#fff',
                      padding: '3px 10px',
                      borderRadius: 4,
                      fontSize: 11,
                      fontWeight: 800,
                      letterSpacing: '0.05em',
                    }}
                  >
                    💣 C4 PLANTED ({frame.extra?.retake_odds ? `Retake Odds: ${Math.round(frame.extra.retake_odds * 100)}%` : 'RETENTION'})
                  </div>
                ) : (
                  <div
                    style={{
                      background: 'rgba(30, 41, 59, 0.7)',
                      border: '1px solid rgba(56, 189, 248, 0.3)',
                      padding: '3px 10px',
                      borderRadius: 4,
                      fontSize: 11,
                      color: '#38bdf8',
                      fontWeight: 600,
                    }}
                  >
                    TARGET: <strong>{frame?.extra?.predicted_site ?? 'SITE A'}</strong> ({Math.round((frame?.extra?.site_prob_a ?? 0.6) * 100)}% Push)
                  </div>
                )}
              </div>

              {/* CT Team side */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right' }}>
                  Buy: <span style={{ color: '#38bdf8' }}>{frame?.extra?.economy_tier_ct ?? 'FULL BUY'}</span> | Alive:{' '}
                  <strong style={{ color: '#f8fafc' }}>{frame?.def_alive ?? 5}</strong>
                </div>
                <span className="font-mono" style={{ fontSize: 20, fontWeight: 800, color: '#06b6d4' }}>
                  {100 - winPct}%
                </span>
                <span
                  style={{
                    background: '#06b6d4',
                    color: '#000',
                    fontWeight: 900,
                    fontSize: 11,
                    padding: '2px 7px',
                    borderRadius: 4,
                  }}
                >
                  COUNTER-TERRORISTS
                </span>
              </div>
            </div>

            {/* Duel Progress Bar */}
            <div
              style={{
                display: 'flex',
                height: 12,
                borderRadius: 6,
                overflow: 'hidden',
                background: '#131b2e',
                boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.6)',
              }}
            >
              <div
                style={{
                  width: `${winPct}%`,
                  background: 'linear-gradient(90deg, #ea580c 0%, #f97316 100%)',
                  transition: 'width 0.15s ease-out',
                }}
              />
              <div
                style={{
                  width: `${100 - winPct}%`,
                  background: 'linear-gradient(90deg, #06b6d4 0%, #0891b2 100%)',
                  transition: 'width 0.15s ease-out',
                }}
              />
            </div>
          </div>

          {/* Radar Screen & Toolbar Container */}
          <div className="glass-panel" style={{ padding: 16, position: 'relative' }}>
            {/* Overlay Layer Toggles */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                marginBottom: 12,
                flexWrap: 'wrap',
              }}
            >
              <span style={{ fontSize: 11, fontWeight: 800, color: '#64748b', marginRight: 4 }}>
                HUD LAYERS:
              </span>
              <div className={`toggle-chip ${showDanger ? 'active' : ''}`} onClick={() => setShowDanger(!showDanger)}>
                🔥 Threat Heatmap
              </div>
              <div className={`toggle-chip ${showSmokes ? 'active' : ''}`} onClick={() => setShowSmokes(!showSmokes)}>
                ☁ Smoke Clouds
              </div>
              <div className={`toggle-chip ${showDeaths ? 'active' : ''}`} onClick={() => setShowDeaths(!showDeaths)}>
                ☠ Death Markers
              </div>
              <div className={`toggle-chip ${showSightlines ? 'active' : ''}`} onClick={() => setShowSightlines(!showSightlines)}>
                ⚡ Sightlines (LOS)
              </div>
              <div className={`toggle-chip ${showCones ? 'active' : ''}`} onClick={() => setShowCones(!showCones)}>
                👁 FOV Vision Cones
              </div>
              <div className={`toggle-chip ${showWalls ? 'active' : ''}`} onClick={() => setShowWalls(!showWalls)}>
                🧱 Walls
              </div>
              <div className={`toggle-chip ${showTerritory ? 'active' : ''}`} onClick={() => setShowTerritory(!showTerritory)}>
                🌐 Voronoi Control
              </div>
            </div>

            {/* Radar Canvas with interactive HUD overlays */}
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
                  border: '1px solid rgba(51, 65, 85, 0.7)',
                  background: '#040711',
                  cursor: 'crosshair',
                  boxShadow: '0 8px 30px rgba(0,0,0,0.8), inset 0 0 50px rgba(0,0,0,0.85)',
                }}
              />

              {/* Real-Time Killfeed Banner in top-right of canvas */}
              {frame?.recent_kills && frame.recent_kills.length > 0 && (
                <div
                  style={{
                    position: 'absolute',
                    top: 18,
                    right: 18,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                    zIndex: 20,
                  }}
                >
                  {frame.recent_kills.map((k, idx) => (
                    <div
                      key={idx}
                      className="killfeed-item font-mono"
                      style={{
                        background: 'rgba(15, 23, 42, 0.9)',
                        border: '1px solid #ef4444',
                        borderRadius: 6,
                        padding: '6px 12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        fontSize: 12,
                        boxShadow: '0 4px 16px rgba(239, 68, 68, 0.4)',
                      }}
                    >
                      <span style={{ color: k.team === 0 ? '#06b6d4' : '#f97316', fontWeight: 700 }}>
                        {k.killer >= 0 ? `P${k.killer + 1}` : 'FRAG'}
                      </span>
                      <span style={{ color: '#f59e0b', fontSize: 13 }}>⚡</span>
                      <span style={{ color: k.team === 0 ? '#f97316' : '#06b6d4', fontWeight: 700 }}>
                        P{k.slot + 1}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Clutch Situation Alert Banner */}
              {frame?.extra?.clutch && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: 20,
                    left: 20,
                    background: 'rgba(15, 23, 42, 0.92)',
                    border: '1px solid #f59e0b',
                    borderRadius: 6,
                    padding: '6px 14px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    fontSize: 12,
                    boxShadow: '0 4px 14px rgba(245, 158, 11, 0.3)',
                    zIndex: 15,
                  }}
                >
                  <span style={{ color: '#f59e0b', fontWeight: 800 }}>⚡ CLUTCH SCENARIO:</span>
                  <span style={{ color: '#f8fafc' }}>
                    {frame.extra.clutch.clutcher} in {frame.extra.clutch.situation} (Odds: {Math.round(frame.extra.clutch.odds * 100)}%)
                  </span>
                </div>
              )}
            </div>

            {/* Replay Timeline Scrubber */}
            <div style={{ marginTop: 14 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: 11,
                  color: 'var(--text-muted)',
                  marginBottom: 6,
                }}
                className="font-mono"
              >
                <span>
                  ROUND TIME: <strong style={{ color: '#f8fafc', fontSize: 13 }}>{frame ? `${frame.clock.toFixed(1)}s` : '0.0s'}</strong>
                </span>
                <span>
                  FRAME: <strong style={{ color: '#38bdf8' }}>{frame ? frame.frame_idx : 0}</strong> / {frame ? frame.total_frames : 0}
                </span>
                <span>
                  SERVER TICK: <strong style={{ color: '#f8fafc' }}>{frame ? frame.tick : '-'}</strong>
                </span>
              </div>

              <div style={{ position: 'relative', width: '100%', height: 20, display: 'flex', alignItems: 'center' }}>
                <input
                  type="range"
                  min={0}
                  max={frame?.total_frames ? frame.total_frames - 1 : 100}
                  value={frame ? frame.frame_idx : 0}
                  onChange={(e) => handleSeek(Number(e.target.value))}
                  className="timeline-slider"
                  title="Drag slider to scrub through round"
                />
              </div>
            </div>
          </div>

          {/* Comprehensive 10-Player Telemetry Matrix */}
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
              <h2 style={{ fontSize: 14, fontWeight: 800, margin: 0, letterSpacing: '0.04em' }}>
                LIVE OPERATOR TELEMETRY ROSTER ({alivePlayers.length} / 10 ACTIVE)
              </h2>
              <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                Click row to inspect personal telemetry & AI threat factors
              </span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: 'var(--text-dim)', textAlign: 'left', borderBottom: '1px solid #334155', fontSize: 11 }}>
                    <th style={{ padding: '6px 8px' }}>OPERATOR</th>
                    <th style={{ padding: '6px 8px' }}>HEALTH</th>
                    <th style={{ padding: '6px 8px' }}>ARMOR</th>
                    <th style={{ padding: '6px 8px' }}>SPEED</th>
                    <th style={{ padding: '6px 8px' }}>TACTICAL COVER</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>AMBUSH DANGER</th>
                  </tr>
                </thead>
                <tbody>
                  {(frame?.players ?? []).map((p) => {
                    const isSelected = selectedSlot === p.slot;
                    const isHovered = hoverSlot === p.slot;
                    const badge = threatBadge(p.threat);
                    return (
                      <tr
                        key={p.slot}
                        onClick={() => setSelectedSlot(p.slot === selectedSlot ? null : p.slot)}
                        onMouseEnter={() => setHoverSlot(p.slot)}
                        onMouseLeave={() => setHoverSlot(null)}
                        style={{
                          cursor: 'pointer',
                          background: isSelected
                            ? 'rgba(56, 189, 248, 0.18)'
                            : isHovered
                            ? 'rgba(255, 255, 255, 0.05)'
                            : 'transparent',
                          opacity: p.alive ? 1 : 0.35,
                          borderBottom: '1px solid rgba(51, 65, 85, 0.3)',
                          transition: 'background 0.15s',
                        }}
                      >
                        <td style={{ padding: '7px 8px', fontWeight: 600 }}>
                          <span style={{ color: p.team === 0 ? '#f97316' : '#06b6d4', marginRight: 6 }}>●</span>
                          P{p.slot + 1} ({p.team === 0 ? 'T' : 'CT'})
                        </td>
                        <td style={{ padding: '7px 8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div style={{ width: 48, height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
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
                          {Math.round(p.armor * 100)}%
                        </td>
                        <td style={{ padding: '7px 8px' }} className="font-mono">
                          {p.speed > 0 ? `${Math.round(p.speed)} u/s` : '0 u/s'}
                        </td>
                        <td style={{ padding: '7px 8px' }}>
                          {p.is_isolated === 1 ? (
                            <span style={{ color: '#ef4444', fontSize: 10, fontWeight: 700, background: 'rgba(239,68,68,0.15)', padding: '2px 6px', borderRadius: 3 }}>
                              ISOLATED
                            </span>
                          ) : p.has_crossfire === 1 ? (
                            <span style={{ color: '#10b981', fontSize: 10, fontWeight: 700, background: 'rgba(16,185,129,0.15)', padding: '2px 6px', borderRadius: 3 }}>
                              CROSSFIRE
                            </span>
                          ) : (
                            <span style={{ color: '#94a3b8', fontSize: 10 }}>SUPPORTED</span>
                          )}
                        </td>
                        <td style={{ padding: '7px 8px', textAlign: 'right' }}>
                          <span
                            className="font-mono"
                            style={{
                              padding: '2px 8px',
                              borderRadius: 4,
                              fontSize: 11,
                              fontWeight: 800,
                              background: badge.bg,
                              color: badge.color,
                              border: `1px solid ${badge.border}`,
                            }}
                          >
                            {Math.round(p.threat * 100)}% ({badge.text})
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

        {/* Right Column: AI Coach Intelligence, Predictive Metrics & Deep Inspector */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Detailed Player Inspector Card */}
          <div className="glass-panel" style={{ padding: 20 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid var(--border-color)',
                paddingBottom: 10,
                marginBottom: 14,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 18 }}>🎯</span>
                <h2 style={{ fontSize: 14, fontWeight: 800, margin: 0, letterSpacing: '0.04em' }}>
                  OPERATOR TELEMETRY INSPECTOR
                </h2>
              </div>
              {inspectedPlayer && (
                <div
                  style={{
                    padding: '3px 9px',
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: 800,
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
                {/* Header info */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
                  <div
                    style={{
                      width: 42,
                      height: 42,
                      borderRadius: 8,
                      background: inspectedPlayer.team === 0 ? '#f97316' : '#06b6d4',
                      color: '#000',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontWeight: 900,
                      fontSize: 17,
                      boxShadow: inspectedPlayer.team === 0 ? '0 0 14px rgba(249,115,22,0.4)' : '0 0 14px rgba(6,182,212,0.4)',
                    }}
                  >
                    P{inspectedPlayer.slot + 1}
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 14 }}>
                      Operator #{inspectedPlayer.slot + 1} ({inspectedPlayer.team === 0 ? 'Terrorist' : 'Counter-Terrorist'})
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }} className="font-mono">
                      X: {inspectedPlayer.x.toFixed(3)} | Y: {inspectedPlayer.y.toFixed(3)} | Yaw: {inspectedPlayer.yaw}°
                    </div>
                  </div>
                </div>

                {/* Micro Stat Grid (4-metrics) */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 14 }}>
                  <div className="stat-box">
                    <span className="stat-label">SPEED</span>
                    <span className="stat-value font-mono">{Math.round(inspectedPlayer.speed)} u/s</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">FIREPOWER</span>
                    <span className="stat-value font-mono">{(inspectedPlayer.weapon_dps * 100).toFixed(0)} DPS</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">TACTICAL STATUS</span>
                    <span className="stat-value" style={{ fontSize: 11, color: inspectedPlayer.is_isolated ? '#ef4444' : '#10b981' }}>
                      {inspectedPlayer.is_isolated ? 'ISOLATED' : inspectedPlayer.has_crossfire ? 'CROSSFIRE' : 'SUPPORTED'}
                    </span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">NEAREST ENEMY</span>
                    <span className="stat-value font-mono" style={{ fontSize: 12 }}>
                      {inspectedPlayer.dist_enemy && inspectedPlayer.dist_enemy > 0
                        ? `${Math.round(inspectedPlayer.dist_enemy)} u (~${(inspectedPlayer.dist_enemy * 0.0254).toFixed(1)}m)`
                        : 'UNKNOWN'}
                    </span>
                  </div>
                </div>

                {/* Paced Coach Threat Diagnostic Sentence */}
                <div
                  style={{
                    background: 'rgba(5, 8, 17, 0.75)',
                    border: '1px solid rgba(56, 189, 248, 0.25)',
                    borderRadius: 8,
                    padding: 12,
                    marginBottom: 12,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ color: '#38bdf8', fontSize: 11, fontWeight: 800 }}>
                      COACH THREAT ASSESSMENT:
                    </span>
                    <span style={{ fontSize: 10, background: 'rgba(16, 185, 129, 0.2)', color: '#10b981', padding: '2px 6px', borderRadius: 4, border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                      ✓ PACED FOR READABILITY
                    </span>
                  </div>
                  <p style={{ margin: 0, fontSize: 13, lineHeight: '1.5', color: '#e2e8f0' }}>
                    {latchedThreatText || inspectedPlayer.why || 'Operator in safe position.'}
                  </p>
                </div>

                {/* Structured SHAP Threat Impact Factor Bars */}
                <div>
                  <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-dim)', display: 'block', marginBottom: 8 }}>
                    TOP DRIVING RISK FACTORS (SHAP VALUES):
                  </span>
                  {(latchedThreatFactors.length > 0 ? latchedThreatFactors : inspectedPlayer.factors || []).map((f, i) => (
                    <div key={i} style={{ marginBottom: 6 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                        <span style={{ color: '#cbd5e1' }}>{f.text}</span>
                        <span style={{ fontWeight: 700, color: f.up ? '#ef4444' : '#10b981' }}>
                          {f.up ? `+${f.share}% RISK` : `-${f.share}% MITIGATES`}
                        </span>
                      </div>
                      <div style={{ height: 4, borderRadius: 2, background: '#1e293b', overflow: 'hidden' }}>
                        <div
                          style={{
                            width: `${Math.min(100, f.share * 2)}%`,
                            height: '100%',
                            background: f.up ? '#ef4444' : '#10b981',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ padding: 24, textAlign: 'center', color: '#64748b', fontSize: 13 }}>
                Hover or click any operator on the radar to inspect.
              </div>
            )}
          </div>

          {/* Round Momentum & Win Predictor Factor Breakdown */}
          <div className="glass-panel" style={{ padding: 20 }}>
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
              <span style={{ fontSize: 18 }}>⚡</span>
              <h2 style={{ fontSize: 14, fontWeight: 800, margin: 0, letterSpacing: '0.04em' }}>
                ROUND MOMENTUM & DYNAMICS
              </h2>
            </div>

            <div
              style={{
                background: 'rgba(5, 8, 17, 0.75)',
                border: '1px solid rgba(51, 65, 85, 0.5)',
                borderRadius: 8,
                padding: 12,
                fontSize: 13,
                lineHeight: '1.5',
                color: '#cbd5e1',
                marginBottom: 12,
              }}
            >
              {latchedWinText || frame?.why_win || 'Analyzing round progression...'}
            </div>

            {/* Win Factors Breakdown */}
            {(latchedWinFactors.length > 0 ? latchedWinFactors : frame?.win_factors || []).map((wf, idx) => (
              <div key={idx} style={{ marginBottom: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                  <span style={{ color: '#94a3b8' }}>{wf.text}</span>
                  <span style={{ fontWeight: 700, color: wf.up ? '#f97316' : '#06b6d4' }}>
                    {wf.up ? `Favours Attack (${wf.share}%)` : `Favours Defense (${wf.share}%)`}
                  </span>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: '#1e293b', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${Math.min(100, wf.share * 2)}%`,
                      height: '100%',
                      background: wf.up ? '#f97316' : '#06b6d4',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Live Tactical Event Log */}
          <div className="glass-panel" style={{ padding: 18, flex: 1 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid var(--border-color)',
                paddingBottom: 10,
                marginBottom: 10,
              }}
            >
              <h2 style={{ fontSize: 14, fontWeight: 800, margin: 0 }}>
                TACTICAL EVENT FEED
              </h2>
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>Auto-Recorded</span>
            </div>

            <div style={{ maxHeight: 200, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
              {tacticalLogs.length === 0 ? (
                <div style={{ fontSize: 12, color: '#64748b', textAlign: 'center', padding: 12 }}>
                  Awaiting in-round eliminations and objective events...
                </div>
              ) : (
                tacticalLogs.map((log) => (
                  <div
                    key={log.id}
                    style={{
                      background: 'rgba(11, 17, 32, 0.6)',
                      borderLeft: `3px solid ${log.color}`,
                      padding: '5px 10px',
                      borderRadius: 4,
                      fontSize: 11,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                    }}
                  >
                    <span className="font-mono" style={{ color: '#94a3b8' }}>
                      [{log.time}]
                    </span>
                    <span style={{ color: '#f8fafc', fontWeight: 600 }}>{log.text}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// 2D Tactical Radar Canvas Drawing Engine
function drawCanvas(
  ctx: CanvasRenderingContext2D,
  f: Frame | null,
  img: HTMLImageElement | null,
  hover: number | null,
  selected: number | null,
  walls: Wall[],
  killAnimations: { x: number; y: number; startTime: number }[],
  opts: {
    showWalls: boolean;
    showSightlines: boolean;
    showDanger: boolean;
    showTerritory: boolean;
    showCones: boolean;
    showSmokes: boolean;
    showDeaths: boolean;
  }
) {
  ctx.clearRect(0, 0, SIZE, SIZE);

  // 1. Base Map Layer
  if (img && img.complete) {
    ctx.drawImage(img, 0, 0, SIZE, SIZE);
  } else {
    ctx.fillStyle = '#040711';
    ctx.fillRect(0, 0, SIZE, SIZE);
  }

  if (!f) return;

  const now = performance.now();

  // 2. Voronoi Territory Control Overlay
  if (opts.showTerritory && f.territory && f.territory.length > 0) {
    const step = SIZE / f.territory.length;
    f.territory.forEach((row, r) => {
      row.forEach((team, c) => {
        if (team < 0) return;
        ctx.fillStyle = team === 0 ? 'rgba(249, 115, 22, 0.12)' : 'rgba(6, 182, 212, 0.12)';
        ctx.fillRect(c * step, r * step, step + 0.5, step + 0.5);
      });
    });
  }

  // 3. Smooth Radial Ambush Danger Heatmap (Radial gradients instead of blocky tiles)
  if (opts.showDanger) {
    for (const p of f.players) {
      if (!p.alive || p.threat < 0.25) continue;
      const px = p.x * SIZE;
      const py = p.y * SIZE;
      const radius = 60 + p.threat * 40;

      const grad = ctx.createRadialGradient(px, py, 4, px, py, radius);
      grad.addColorStop(0, `rgba(239, 68, 68, ${p.threat * 0.45})`);
      grad.addColorStop(0.6, `rgba(239, 68, 68, ${p.threat * 0.2})`);
      grad.addColorStop(1, 'rgba(239, 68, 68, 0)');

      ctx.beginPath();
      ctx.arc(px, py, radius, 0, 2 * Math.PI);
      ctx.fillStyle = grad;
      ctx.fill();
    }
  }

  // 4. Yellow Obstacle Walls
  if (opts.showWalls && walls && walls.length > 0) {
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#eab308'; // vibrant yellow
    for (const [p1, p2] of walls) {
      ctx.beginPath();
      ctx.moveTo(p1[0] * SIZE, p1[1] * SIZE);
      ctx.lineTo(p2[0] * SIZE, p2[1] * SIZE);
      ctx.stroke();
    }
  }

  // 5. Active Volumetric Smoke Grenades
  if (opts.showSmokes && f.smokes && f.smokes.length > 0) {
    for (const s of f.smokes) {
      const sx = s.x * SIZE;
      const sy = s.y * SIZE;
      const sRadius = (s.r || 0.028) * SIZE * 1.5;

      // Volumetric billowing smoke gradient
      const smokeGrad = ctx.createRadialGradient(sx, sy, 5, sx, sy, sRadius);
      smokeGrad.addColorStop(0, 'rgba(241, 245, 249, 0.7)');
      smokeGrad.addColorStop(0.5, 'rgba(203, 213, 225, 0.45)');
      smokeGrad.addColorStop(1, 'rgba(148, 163, 184, 0)');

      ctx.beginPath();
      ctx.arc(sx, sy, sRadius, 0, 2 * Math.PI);
      ctx.fillStyle = smokeGrad;
      ctx.fill();

      // Outer smoke dissipation progress ring
      ctx.beginPath();
      ctx.arc(sx, sy, sRadius, 0, 2 * Math.PI * (1 - s.progress));
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // 6. Persistent Elimination / Death Markers on Floor
  if (opts.showDeaths && f.deaths && f.deaths.length > 0) {
    for (const d of f.deaths) {
      const dx = d.x * SIZE;
      const dy = d.y * SIZE;
      const teamColor = d.team === 0 ? '#ea580c' : '#0284c7';

      // Draw elegant death X skull icon on floor
      ctx.beginPath();
      ctx.arc(dx, dy, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
      ctx.fill();
      ctx.strokeStyle = teamColor;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Cross lines
      ctx.beginPath();
      ctx.moveTo(dx - 3, dy - 3);
      ctx.lineTo(dx + 3, dy + 3);
      ctx.moveTo(dx + 3, dy - 3);
      ctx.lineTo(dx - 3, dy + 3);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // 7. Active Line-of-Sight Laser Rays
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

  // 8. Animated Kill Shockwaves (Expanding Ripple)
  for (let i = killAnimations.length - 1; i >= 0; i--) {
    const k = killAnimations[i];
    const age = now - k.startTime;
    if (age > 1800) {
      killAnimations.splice(i, 1);
      continue;
    }
    const ripRadius = (age / 1800) * 55;
    const ripAlpha = 1 - age / 1800;

    ctx.beginPath();
    ctx.arc(k.x, k.y, ripRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = `rgba(239, 68, 68, ${ripAlpha})`;
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  // 9. Players, Cones & Threat Rings
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
      ctx.arc(px, py, 58, rad - Math.PI / 4, rad + Math.PI / 4);
      ctx.closePath();
      ctx.fillStyle = p.team === 0 ? 'rgba(249, 115, 22, 0.14)' : 'rgba(6, 182, 212, 0.14)';
      ctx.fill();
    }

    // High Danger Pulsing Ring
    if (p.threat >= 0.6) {
      const pulse = 4 * Math.sin(now / 110);
      ctx.beginPath();
      ctx.arc(px, py, 16 + pulse, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.9)';
      ctx.lineWidth = 3.5;
      ctx.stroke();
    } else if (p.threat >= 0.3) {
      ctx.beginPath();
      ctx.arc(px, py, 15, 0, 2 * Math.PI);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Selected crosshair ring
    if (isTargeted) {
      ctx.beginPath();
      ctx.arc(px, py, 22, 0, 2 * Math.PI);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Core Player Dot
    ctx.beginPath();
    ctx.arc(px, py, 9.5, 0, 2 * Math.PI);
    ctx.globalAlpha = 0.4 + 0.6 * p.hp;
    ctx.fillStyle = p.team === 0 ? '#f97316' : '#06b6d4';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000';
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Operator Number
    ctx.fillStyle = '#000';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${p.slot + 1}`, px, py);
  }
}
