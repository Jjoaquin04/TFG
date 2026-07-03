import json
import os
import cv2
import time
import numpy as np
import config
from ultralytics import YOLO
from core import BallTracker ,KeypointsCourt, PlayerTracker
from utils import read_video, make_prediction_batch, make_track_batch

def extract_seq_b1(url_video):
    
    inicio = time.time()
    # ──────── Variables and initialization ────────────────────────────────────
    cap, fps, width, height  = read_video(url_video)

    court_model = YOLO(config.KEYPOINTS_COURT_MODEL, task='pose')
    player_model = YOLO(config.PLAYER_POSE_MODEL, task='pose')
    ball_model = YOLO(config.BALL_MODEL, task='detect')

    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    keypoints_court = KeypointsCourt()
    # ───────────────────────────────────────────────────────

    is_first_frame = True
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1

        if is_first_frame:
            result_keypoints = make_prediction_batch(court_model, [frame], conf_grade = 0.25, batch_size=1)
            kps_obj = result_keypoints[0].keypoints if isinstance(result_keypoints, list) else result_keypoints.keypoints
            if kps_obj is None or len(kps_obj.xy[0]) < 4:
                continue
                
            kps = kps_obj.xy.cpu().numpy()
            keypoints_court.refine_points(frame, kps[0])
            keypoints_court.extract_rest_of_kpoints()
            is_first_frame = False

        print(f"Pasando batch a los modelos de deteccion\n")
        # Pasamos [frame] para que la función procese una lista de 1 elemento
        # y devuelva una lista de 1 resultado, igual que en multithread
        result_players = make_track_batch(player_model, [frame], batch_size=1)
        result_ball = make_prediction_batch(ball_model, [frame], conf_grade = 0.40, batch_size=1)

        res_players = result_players[0]
        res_ball = result_ball[0]
    
        boxes, ids, keypoints = [], [], None
        if res_players.boxes is not None and len(res_players.boxes) > 0 and res_players.boxes.id is not None:
            boxes = res_players.boxes.xyxy.cpu().numpy() 
            ids = res_players.boxes.id.cpu().numpy().astype(int)
            keypoints = res_players.keypoints.data.cpu().numpy() if hasattr(res_players, 'keypoints') and res_players.keypoints is not None else None

        player_tracker.update(ids, boxes, keypoints, frame_idx)
    
        if hasattr(res_ball, 'boxes') and res_ball.boxes is not None:
            ball_boxes = res_ball.boxes.xyxy.cpu().numpy()
            if len(ball_boxes) > 0:
                ball_tracker.update(ball_boxes[0], frame_idx) 
            else:
                ball_tracker.update(None, frame_idx)
        else:
            ball_tracker.update(None, frame_idx)
        
    cap.release()   
    ball_history = ball_tracker.get_ball_history()
    players_history = player_tracker.get_players_history()
    court_information = keypoints_court.get_court_information()

    detection = {
        'ball': ball_history,
        'players': players_history,
        'court': court_information
    }

    os.makedirs(config.RAW_JSON_FOLDER_PATH, exist_ok=True)
    video_name = os.path.basename(url_video).split('.')[0]
    output_path = os.path.join(config.RAW_JSON_FOLDER_PATH, f"{video_name}_seq_b1.json")
    with open(output_path, 'w') as f:
        json.dump(detection, f, cls=config.NumpyEncoder, indent=2)

    fin = time.time()   
    total = fin - inicio
    print(f"Tiempo -> {total:.6f} segundos")
    print(f"Extraccion finalizada! Raw Data guardada en {output_path}")
