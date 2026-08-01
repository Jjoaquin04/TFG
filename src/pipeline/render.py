import cv2
import numpy as np
import json
import os
from tqdm import tqdm
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
    players_metadata = data.get('players_metadata', {})
    events_data_by_frame = data.get('events', {})

    court_info = data.get('court', None)
    court_keypoints = None
    if court_info is not None and len(court_info) > 0:
        court_keypoints = np.array(court_info[0], dtype=np.float32).reshape(-1, 1, 2)
        
    cap, frame_height, frame_width, fps = read_video(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_path = os.path.join('data', 'outputs', 'videos', output_video_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(int(round(frame_width)), int(round(frame_height))))

    mini_court = MiniCourt(frame_width)
    
    frame_idx = 1
    current_hit_line = None
    current_event_text = []

    print("Starting renderization process...")
    pbar = tqdm(total=total_frames, desc="Rendering Frames")
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
            
        if court_keypoints is not None:
            draw_edges_court_connections(img, court_keypoints)
            
        ball_info = ball_data_by_frame.get(frame_idx)
        ball_detected = False
        if ball_info is not None:
            x_min = ball_info.get('x_min')
            if x_min is not None and not np.isnan(x_min):
                ball_detected = True

        if ball_detected:
            bbx = [ball_info['x_min'], ball_info['y_min'], ball_info['x_max'], ball_info['y_max']]
            draw_bounding_boxes(img, [bbx])

        players_bbx = []
        players_ids = []
        players_positions = []
        for player_id, player_frames in players_data_by_frame.items():
            p = player_frames.get(str(frame_idx))
            if p and p.get('x_min') is not None:
                bbx = [p['x_min'], p['y_min'], p['x_max'], p['y_max']]
                players_bbx.append(bbx)
                players_ids.append(p.get('player_id', 0))
                if 'real_x' in p and p['real_x'] is not None:
                    players_positions.append([p['real_x'], p['real_y']])
        
        if players_bbx:
            draw_bounding_boxes(img, players_bbx, players_ids)

        events_info = events_data_by_frame.get(str(frame_idx), [])
        if events_info:
            event = events_info[0]
            current_hit_line = (event.get('origin_cord'), event.get('destiny_cord'))
            
            stroke = event.get('type_of_shot', 'HIT')
            if stroke is None: stroke = 'HIT'
                
            player = event.get('player_id', '?')
            impact = event.get('impact_frame', '?')
            traj = event.get('trajectory', 'N/A')
            
            player_metadata = players_metadata.get(str(player), {})
            hand = player_metadata.get('racket_hand', "Unknown")

            current_event_text = [
                f"Impact Frame: {impact}",
                f"Player: {player} ({hand} hand)",
                f"Shot: {stroke.upper()}",
                f"Trajectory: {traj}"
            ]
            
        if current_event_text:
            y_offset = 60
            for line in current_event_text:
                cv2.putText(img, line, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_offset += 30

        img = mini_court.draw_court(img, players_positions, current_hit_line)
        out.write(img)
        frame_idx += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    out.release()
    print(f"\nRender finished! Video saved in {out_path}")