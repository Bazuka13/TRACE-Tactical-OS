"""backend/generate_sample_video.py
Generates an automated, realistic CS2 broadcast match gameplay video
and saves it to data/videos/sample_gameplay.mp4.

Features:
- Full 1280x720 @ 30 FPS video with CS2 spectator broadcast HUD.
- Sub-sampled 220x220 Mirage radar minimap in top-left with real de_mirage.png.
- Color-accurate Orange T (B=20, G=120, R=245) and Cyan CT (B=220, G=180, R=15) dots
  matching the exact HSV/RGB color thresholds in cv_radar_engine.py.
- Dynamic match events: smoke grenade on stairs, bomb plant at A-Site,
  and real-time kill feed updates in top-right with kill animations.
"""

import os
import math
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / 'data' / 'videos'
MAPS_DIR = ROOT / 'data' / 'maps'
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = VIDEOS_DIR / 'sample_gameplay.mp4'

def create_gameplay_video(filename=OUTPUT_FILE, duration_sec=12, fps=30):
    width, height = 1280, 720
    total_frames = int(duration_sec * fps)
    
    # Load Mirage Map Overview
    map_path = MAPS_DIR / 'de_mirage.png'
    if map_path.exists():
        raw_map = cv2.imread(str(map_path))
    else:
        # Fallback dark tactical grid
        raw_map = np.full((1024, 1024, 3), 25, dtype=np.uint8)
        for g in range(0, 1024, 64):
            cv2.line(raw_map, (g, 0), (g, 1024), (45, 45, 45), 1)
            cv2.line(raw_map, (0, g), (1024, g), (45, 45, 45), 1)

    minimap_size = 220
    minimap_base = cv2.resize(raw_map, (minimap_size, minimap_size), interpolation=cv2.INTER_AREA)

    # Use mp4v fourcc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(filename), fourcc, float(fps), (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {filename}")

    # Player Trajectory Definitions (Normalized 0.0 to 1.0 on radar)
    # T Players (Orange): Target Site A
    t_players = [
        {"name": "s1mple", "start": (0.30, 0.66), "end": (0.41, 0.63), "alive": True, "death_frame": None},
        {"name": "b1t",    "start": (0.28, 0.69), "end": (0.39, 0.65), "alive": True, "death_frame": None},
        {"name": "jL",     "start": (0.33, 0.74), "end": (0.42, 0.69), "alive": True, "death_frame": 260},
        {"name": "iM",     "start": (0.32, 0.54), "end": (0.40, 0.58), "alive": True, "death_frame": None},
        {"name": "Aleksib","start": (0.41, 0.50), "end": (0.45, 0.55), "alive": True, "death_frame": None},
    ]

    # CT Players (Cyan): Defense Site A & Mid
    ct_players = [
        {"name": "karrigan","start": (0.50, 0.65), "end": (0.48, 0.64), "alive": True, "death_frame": 140},
        {"name": "frozen",  "start": (0.45, 0.60), "end": (0.44, 0.61), "alive": True, "death_frame": 200},
        {"name": "broky",   "start": (0.43, 0.67), "end": (0.42, 0.66), "alive": True, "death_frame": None},
        {"name": "ropz",    "start": (0.47, 0.48), "end": (0.46, 0.52), "alive": True, "death_frame": None},
        {"name": "rain",    "start": (0.60, 0.40), "end": (0.49, 0.54), "alive": True, "death_frame": None},
    ]

    print(f"[*] Rendering {total_frames} frames ({duration_sec}s @ {fps}fps) to {filename}...")

    for f_idx in range(total_frames):
        alpha = f_idx / max(1, total_frames - 1)
        sim_time_s = f_idx / fps
        clock_remain = max(0, 75 - int(sim_time_s))
        mins, secs = divmod(clock_remain, 60)
        clock_str = f"{mins}:{secs:02d}"

        # -------------------------------------------------------------
        # 1. Main Viewport: Stylized CS2 Desert Perspective Backdrop
        # -------------------------------------------------------------
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky and ground gradients (Mirage A-Site daytime atmosphere)
        for y in range(height):
            if y < 420:
                # Sky to horizon gradient (dusty golden-blue atmosphere)
                t_y = y / 420.0
                b = int(140 * (1 - t_y) + 70 * t_y)
                g = int(100 * (1 - t_y) + 90 * t_y)
                r = int(70 * (1 - t_y) + 130 * t_y)
                canvas[y, :] = (b, g, r)
            else:
                # Sandstone ground perspective gradient
                t_y = (y - 420) / (height - 420)
                b = int(50 * (1 - t_y) + 25 * t_y)
                g = int(75 * (1 - t_y) + 45 * t_y)
                r = int(115 * (1 - t_y) + 70 * t_y)
                canvas[y, :] = (b, g, r)

        # 3D Architecture: Mirage A-Site Pillars, Ramp & Palace
        # Left wall (A-Ramp stone wall)
        pts_left_wall = np.array([[0, 200], [380, 320], [380, 580], [0, 720]], np.int32)
        cv2.fillPoly(canvas, [pts_left_wall], (42, 68, 105))
        cv2.polylines(canvas, [pts_left_wall], True, (60, 95, 145), 2)

        # Right building (Palace balcony & entrance)
        pts_right_bldg = np.array([[880, 280], [1280, 160], [1280, 720], [880, 620]], np.int32)
        cv2.fillPoly(canvas, [pts_right_bldg], (38, 62, 98))
        cv2.polylines(canvas, [pts_right_bldg], True, (55, 88, 138), 2)

        # Center A-Site boxes / default plant structure
        box_center_x = 640 + int(20 * math.sin(sim_time_s * 0.8))
        cv2.rectangle(canvas, (box_center_x - 70, 380), (box_center_x + 70, 470), (32, 50, 78), -1)
        cv2.rectangle(canvas, (box_center_x - 70, 380), (box_center_x + 70, 470), (75, 115, 165), 2)

        # Stairs structure
        cv2.line(canvas, (380, 520), (520, 460), (70, 105, 155), 3)
        cv2.line(canvas, (380, 540), (520, 480), (60, 90, 135), 2)

        # Smoke grenade deployment at CT Stairs (after frame 90)
        if f_idx >= 90:
            smoke_age = min(1.0, (f_idx - 90) / 45.0)
            smoke_radius = int(80 * smoke_age)
            smoke_overlay = canvas.copy()
            cv2.circle(smoke_overlay, (510, 450), smoke_radius, (170, 175, 180), -1)
            cv2.circle(smoke_overlay, (540, 430), int(smoke_radius * 0.8), (190, 195, 200), -1)
            cv2.circle(smoke_overlay, (480, 460), int(smoke_radius * 0.7), (150, 155, 160), -1)
            cv2.addWeighted(smoke_overlay, 0.75, canvas, 0.25, 0, canvas)

        # First-Person Crosshair (Center)
        cx, cy = 640, 360
        ch_len = 10
        ch_gap = 4
        # Dynamic recoil / sway
        sway_x = int(3 * math.sin(f_idx * 0.2))
        sway_y = int(2 * math.cos(f_idx * 0.2))
        cv2.line(canvas, (cx - ch_len - ch_gap + sway_x, cy + sway_y), (cx - ch_gap + sway_x, cy + sway_y), (0, 255, 0), 2)
        cv2.line(canvas, (cx + ch_gap + sway_x, cy + sway_y), (cx + ch_len + ch_gap + sway_x, cy + sway_y), (0, 255, 0), 2)
        cv2.line(canvas, (cx + sway_x, cy - ch_len - ch_gap + sway_y), (cx + sway_x, cy - ch_gap + sway_y), (0, 255, 0), 2)
        cv2.line(canvas, (cx + sway_x, cy + ch_gap + sway_y), (cx + sway_x, cy + ch_len + ch_gap + sway_y), (0, 255, 0), 2)
        cv2.circle(canvas, (cx + sway_x, cy + sway_y), 1, (0, 255, 0), -1)

        # Gun Viewmodel in Lower Right (AK-47 barrel silhouette)
        gun_pts = np.array([
            [960, 720], [1020, 520], [1060, 510], [1080, 535],
            [1140, 545], [1180, 520], [1280, 530], [1280, 720]
        ], np.int32)
        cv2.fillPoly(canvas, [gun_pts], (22, 22, 26))
        cv2.polylines(canvas, [gun_pts], False, (65, 65, 75), 2)

        # -------------------------------------------------------------
        # 2. Minimap Radar in Top-Left (220x220 @ x=25, y=25)
        # -------------------------------------------------------------
        mx, my = 25, 25
        radar_view = minimap_base.copy()

        # Circular radar bezel mask
        mask = np.zeros((minimap_size, minimap_size), dtype=np.uint8)
        cv2.circle(mask, (minimap_size // 2, minimap_size // 2), minimap_size // 2 - 2, 255, -1)

        # Draw Smoke on Radar
        if f_idx >= 90:
            smoke_rx = int(0.45 * minimap_size)
            smoke_ry = int(0.60 * minimap_size)
            cv2.circle(radar_view, (smoke_rx, smoke_ry), 18, (160, 160, 160), -1)

        # Draw Bomb Plant on Radar (at frame 200+)
        if f_idx >= 200:
            bx = int(0.43 * minimap_size)
            by = int(0.66 * minimap_size)
            pulse = int(3 * math.sin(f_idx * 0.4))
            cv2.circle(radar_view, (bx, by), 7 + pulse, (0, 0, 255), -1)
            cv2.putText(radar_view, "C4", (bx - 5, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Render Active T Players (Pure Orange: B=20, G=120, R=245)
        # Checked by cv_radar_engine: R > 175, 60 <= G <= 175, B < 95
        t_alive_cnt = 0
        for p in t_players:
            if p["death_frame"] and f_idx >= p["death_frame"]:
                continue
            t_alive_cnt += 1
            # Current coordinate interpolation
            px = int((p["start"][0] + (p["end"][0] - p["start"][0]) * alpha) * minimap_size)
            py = int((p["start"][1] + (p["end"][1] - p["start"][1]) * alpha) * minimap_size)
            
            # Subtle micro-jitter to simulate walking
            px += int(1.5 * math.sin(f_idx * 0.3 + hash(p["name"]) % 5))
            py += int(1.5 * math.cos(f_idx * 0.3 + hash(p["name"]) % 5))

            # Draw Terrorist Dot
            cv2.circle(radar_view, (px, py), 6, (20, 120, 245), -1)
            cv2.circle(radar_view, (px, py), 7, (0, 0, 0), 1)

        # Render Active CT Players (Pure Cyan: B=220, G=180, R=15)
        # Checked by cv_radar_engine: B > 155, G > 115, R < 110
        ct_alive_cnt = 0
        for p in ct_players:
            if p["death_frame"] and f_idx >= p["death_frame"]:
                continue
            ct_alive_cnt += 1
            px = int((p["start"][0] + (p["end"][0] - p["start"][0]) * alpha) * minimap_size)
            py = int((p["start"][1] + (p["end"][1] - p["start"][1]) * alpha) * minimap_size)
            
            px += int(1.0 * math.cos(f_idx * 0.25 + hash(p["name"]) % 5))
            py += int(1.0 * math.sin(f_idx * 0.25 + hash(p["name"]) % 5))

            # Draw Counter-Terrorist Dot
            cv2.circle(radar_view, (px, py), 6, (220, 180, 15), -1)
            cv2.circle(radar_view, (px, py), 7, (0, 0, 0), 1)

        # Draw Radar Circular Sweep Line
        sweep_rad = (f_idx * 0.08) % (2 * math.pi)
        sw_end_x = int(minimap_size // 2 + (minimap_size // 2 - 4) * math.cos(sweep_rad))
        sw_end_y = int(minimap_size // 2 + (minimap_size // 2 - 4) * math.sin(sweep_rad))
        cv2.line(radar_view, (minimap_size // 2, minimap_size // 2), (sw_end_x, sw_end_y), (80, 220, 120), 1)

        # Apply circular bezel
        radar_masked = cv2.bitwise_and(radar_view, radar_view, mask=mask)
        canvas_roi = canvas[my:my+minimap_size, mx:mx+minimap_size]
        inv_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(canvas_roi, canvas_roi, mask=inv_mask)
        canvas[my:my+minimap_size, mx:mx+minimap_size] = cv2.add(bg, radar_masked)

        # Radar Outer Chrome Ring
        cv2.circle(canvas, (mx + minimap_size // 2, my + minimap_size // 2), minimap_size // 2, (56, 189, 248), 2)
        cv2.putText(canvas, "MIRAGE - A SITE", (mx + 30, my + minimap_size + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1)

        # -------------------------------------------------------------
        # 3. Top Spectator Scoreboard HUD
        # -------------------------------------------------------------
        sb_x, sb_y, sb_w, sb_h = 470, 20, 340, 54
        # Semi-transparent dark glass panel
        sb_roi = canvas[sb_y:sb_y+sb_h, sb_x:sb_x+sb_w]
        sb_bg = np.full_like(sb_roi, (12, 16, 28))
        cv2.addWeighted(sb_bg, 0.88, sb_roi, 0.12, 0, canvas[sb_y:sb_y+sb_h, sb_x:sb_x+sb_w])
        cv2.rectangle(canvas, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h), (50, 65, 95), 1)

        # Team T Score & Name (NAVI, 11)
        cv2.rectangle(canvas, (sb_x + 6, sb_y + 8), (sb_x + 95, sb_y + 46), (15, 75, 180), -1)
        cv2.putText(canvas, "NAVI", (sb_x + 14, sb_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(canvas, "11", (sb_x + 68, sb_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Center Clock
        cv2.putText(canvas, clock_str, (sb_x + 138, sb_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Team CT Score & Name (FAZE, 9)
        cv2.rectangle(canvas, (sb_x + sb_w - 95, sb_y + 8), (sb_x + sb_w - 6, sb_y + 46), (140, 115, 15), -1)
        cv2.putText(canvas, "FAZE", (sb_x + sb_w - 90, sb_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(canvas, "9", (sb_x + sb_w - 36, sb_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Alive Pips below scoreboard
        for i in range(5):
            col_t = (20, 120, 245) if i < t_alive_cnt else (50, 50, 60)
            col_ct = (220, 180, 15) if i < ct_alive_cnt else (50, 50, 60)
            cv2.rectangle(canvas, (sb_x + 10 + i * 16, sb_y + sb_h + 3), (sb_x + 22 + i * 16, sb_y + sb_h + 7), col_t, -1)
            cv2.rectangle(canvas, (sb_x + sb_w - 22 - i * 16, sb_y + sb_h + 3), (sb_x + sb_w - 10 - i * 16, sb_y + sb_h + 7), col_ct, -1)

        # -------------------------------------------------------------
        # 4. Top-Right Broadcast Kill Feed
        # -------------------------------------------------------------
        kf_x, kf_y = 960, 25
        events = [
            (140, "NAVI b1t", "AK-47", "FAZE karrigan"),
            (200, "NAVI s1mple", "AWP", "FAZE frozen"),
            (260, "FAZE ropz", "M4A1-S", "NAVI jL")
        ]
        curr_y = kf_y
        for trig_f, killer, weapon, victim in events:
            if f_idx >= trig_f:
                age = f_idx - trig_f
                if age < 100:
                    kf_w, kf_h = 295, 28
                    cv2.rectangle(canvas, (kf_x, curr_y), (kf_x + kf_w, curr_y + kf_h), (10, 15, 25), -1)
                    cv2.rectangle(canvas, (kf_x, curr_y), (kf_x + kf_w, curr_y + kf_h), (239, 68, 68), 1)
                    cv2.putText(canvas, killer, (kf_x + 8, curr_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (249, 115, 22) if "NAVI" in killer else (6, 182, 212), 1)
                    cv2.putText(canvas, f"[{weapon}]", (kf_x + 115, curr_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 205), 1)
                    cv2.putText(canvas, victim, (kf_x + 195, curr_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (6, 182, 212) if "FAZE" in victim else (249, 115, 22), 1)
                    curr_y += 34

        # -------------------------------------------------------------
        # 5. Center Alert Banner (Bomb Planted at frame 200)
        # -------------------------------------------------------------
        if 200 <= f_idx <= 270:
            banner_w, banner_h = 420, 36
            bx0 = (width - banner_w) // 2
            by0 = 120
            cv2.rectangle(canvas, (bx0, by0), (bx0 + banner_w, by0 + banner_h), (15, 15, 30), -1)
            cv2.rectangle(canvas, (bx0, by0), (bx0 + banner_w, by0 + banner_h), (239, 68, 68), 2)
            cv2.putText(canvas, "⚠ THE BOMB HAS BEEN PLANTED", (bx0 + 28, by0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2)

        # -------------------------------------------------------------
        # 6. Bottom First-Person Tactical HUD
        # -------------------------------------------------------------
        # Health & Armor on Left
        cv2.rectangle(canvas, (35, height - 60), (230, height - 20), (12, 16, 28), -1)
        cv2.rectangle(canvas, (35, height - 60), (230, height - 20), (50, 65, 95), 1)
        cv2.putText(canvas, "+ 100", (48, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(canvas, "HP", (110, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (148, 163, 184), 1)
        cv2.putText(canvas, "🛡 100", (145, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)

        # Weapon & Ammo on Right
        cv2.rectangle(canvas, (width - 240, height - 60), (width - 35, height - 20), (12, 16, 28), -1)
        cv2.rectangle(canvas, (width - 240, height - 60), (width - 35, height - 20), (50, 65, 95), 1)
        ammo_count = max(18, 30 - int((f_idx // 30) * 2))
        cv2.putText(canvas, f"{ammo_count} / 90", (width - 220, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(canvas, "AK-47", (width - 110, height - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (249, 115, 22), 1)

        out.write(canvas)

    out.release()
    file_size_mb = os.path.getsize(str(filename)) / (1024 * 1024)
    print(f"[OK] Automated Gameplay Video Generated Successfully!")
    print(f"    File: {filename}")
    print(f"    Size: {file_size_mb:.2f} MB")
    print(f"    Resolution: {width}x{height} @ {fps}fps")
    return {
        "filename": filename.name,
        "path": str(filename),
        "size_mb": round(file_size_mb, 2),
        "fps": fps,
        "frames": total_frames,
        "duration_sec": duration_sec
    }


if __name__ == '__main__':
    create_gameplay_video()
