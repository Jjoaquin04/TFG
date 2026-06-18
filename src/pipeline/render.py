import cv2
import numpy as np
import json
import os
from core import MiniCourt
from utils import read_video, draw_bounding_boxes, draw_edges_court_connections

def render(video_path: str, interp_json_path: str):
   
    video_name = os.path.basename(video_path).split('.')[0]
    output_video_name = f"{video_name}_rendered.mp4"

    print(f"Loading tracking data from {interp_json_path}...")
    with open(interp_json_path, 'r') as f:
        data = json.load(f)
    
    ball_data_by_frame = {item['frame']: item for item in data.get('ball', [])}
    players_data_by_frame = data.get('players', {})
    events_data_by_frame = data.get('events', {})

    # Court keypoints
    court_info = data.get('court', None)
    court_keypoints = None
    if court_info is not None and len(court_info) > 0:
        court_keypoints = np.array(court_info[0], dtype=np.float32).reshape(-1, 1, 2)
        
    # ──────── Iniciar video ────────
    cap, frame_height, frame_width, fps = read_video(video_path)
    out_path = os.path.join('data', 'outputs', 'videos', output_video_name)
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
        
        if court_keypoints is not None:
            draw_edges_court_connections(img, court_keypoints)

        ball_info = ball_data_by_frame.get(frame_idx)
        ball_pos = None
        
        ball_detected = False
        if ball_info is not None:
            x_min = ball_info.get('x_min')
            if x_min is not None and not np.isnan(x_min):
                ball_detected = True

        if ball_detected:
            bbx = [ball_info['x_min'], ball_info['y_min'], ball_info['x_max'], ball_info['y_max']]
            draw_bounding_boxes(img, [bbx])
            if 'real_x' in ball_info and ball_info['real_x'] is not None and not np.isnan(ball_info['real_x']):
                ball_pos = np.array([[[ball_info['real_x'], ball_info['real_y']]]], dtype=np.float32)
        else:
            cv2.putText(img, f"no deteccion {frame_idx}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        players_info = []
        for player_id, player_frames in players_data_by_frame.items():
            record = player_frames.get(str(frame_idx))
            if record:
                players_info.append(record)
                
        players_bbx = []
        players_ids = []
        players_positions = []
        
        for p in players_info:
            if p.get('x_min') is not None:
                bbx = [p['x_min'], p['y_min'], p['x_max'], p['y_max']]
                players_bbx.append(bbx)
                players_ids.append(p.get('player_id', 0))
                if 'real_x' in p and p['real_x'] is not None:
                    players_positions.append([p['real_x'], p['real_y']])
            
                 
        if players_bbx:
            draw_bounding_boxes(img, players_bbx, players_ids)

        
        # (Mantenemos un contador de frames para que el texto dure visible en pantalla)
        if not hasattr(render, 'hit_counter'):
            render.hit_counter = 0
            render.current_hit_text = ""
            
        events_info = events_data_by_frame.get(str(frame_idx), [])
        if events_info:
            event = events_info[0]
            player_id = str(event['player_id'])
            event_impact_frame = str(event['impact_frame'])
            # Activamos el contador para que dure 10 frames (~0.3 seg)
            render.hit_counter = 10
            render.current_hit_text = f"HIT! (Player {player_id}) - Frame {event_impact_frame}"
            
        if render.hit_counter > 0:
            cv2.putText(img, render.current_hit_text, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            render.hit_counter -= 1

        img_with_minicourt = mini_court.draw_court(img, players_positions, ball_pos)
        out.write(img_with_minicourt)

    cap.release()
    out.release()
    print(f"\nRender finished! Video saved in {out_path}")
