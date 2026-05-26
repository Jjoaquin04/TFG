import cv2
import numpy as np
import json
import os
import config
from core import MiniCourt
from utils import read_video, draw_bounding_boxes, draw_edges_court_connections

class RenderItem:
    """Clase proxy para que las funciones de renderizado puedan leer .bbx de forma transparente."""
    def __init__(self, bbx):
        self.bbx = bbx

def render(json_file_name: str, output_video_name: str = 'output_video.mp4'):
    interp_path = os.path.join(config.INTERP_JSON_PATH, json_file_name)
    
    print(f"Loading tracking data from {interp_path}...")
    with open(interp_path, 'r') as f:
        data = json.load(f)
        
    # Organizar datos por frame para un acceso rápido O(1)
    ball_data_by_frame = {item['frame']: item for item in data.get('ball', [])}
    
    # Si tenemos players en el JSON, los agrupamos también por frame
    players_data_by_frame = {}
    for p in data.get('players', []):
        frame = p['frame']
        if frame not in players_data_by_frame:
            players_data_by_frame[frame] = []
        players_data_by_frame[frame].append(p)
        
    # Court keypoints (si los extrajimos y guardamos en la fase anterior)
    court_keypoints = data.get('court_keypoints', None)
    if court_keypoints is not None:
        court_keypoints = np.array(court_keypoints, dtype=np.float32).reshape(-1, 1, 2)
        
    # ──────── Iniciar video ────────
    cap, frame_height, frame_width, fps = read_video(config.VIDEO_PATH)
    out_path = os.path.join('data/outputs', output_video_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(int(round(frame_width)), int(round(frame_height))))

    mini_court = MiniCourt(frame_width)

    print("Starting renderization process...")
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
            
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Rendering frame {frame_idx} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}", end="\r")
        
        # 1. Dibujar líneas de la pista principal si tenemos los puntos
        if court_keypoints is not None:
            draw_edges_court_connections(img, court_keypoints)

        # 2. Dibujar Pelota
        ball_info = ball_data_by_frame.get(frame_idx)
        ball_pos = None
        if ball_info is not None and not np.isnan(ball_info.get('x_min', np.nan)):
            bbx = [ball_info['x_min'], ball_info['y_min'], ball_info['x_max'], ball_info['y_max']]
            draw_bounding_boxes(img, [RenderItem(bbx)])
            if 'real_x' in ball_info and not np.isnan(ball_info['real_x']):
                ball_pos = np.array([[[ball_info['real_x'], ball_info['real_y']]]], dtype=np.float32)

        # 3. Dibujar Jugadores
        players_info = players_data_by_frame.get(frame_idx, [])
        players_objects = []
        players_ids = []
        players_positions = []
        
        for p in players_info:
            if p.get('x_min') is not None:
                bbx = [p['x_min'], p['y_min'], p['x_max'], p['y_max']]
                players_objects.append(RenderItem(bbx))
                players_ids.append(p.get('player_id', 0))
            if 'real_x' in p and p['real_x'] is not None:
                 players_positions.append([p['real_x'], p['real_y']])
                 
        if players_objects:
            draw_bounding_boxes(img, players_objects, players_ids)

        # 4. Dibujar Mini Court
        img_with_minicourt = mini_court.draw_court(img, players_positions, ball_pos)
        
        # Escribir frame
        out.write(img_with_minicourt)

    cap.release()
    out.release()
    print(f"\nRender finished! Video saved in {out_path}")
