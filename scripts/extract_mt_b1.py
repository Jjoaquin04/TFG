import json
import os
import cv2
import time
import numpy as np
import config
from queue import Queue
from threading import Thread
from ultralytics import YOLO
from core import BallTracker ,KeypointsCourt, PlayerTracker
from utils import read_video, make_prediction_batch, make_track_batch, video_reader

def extract_mt_b1(url_video):
    
    inicio = time.time()
    # ──────── Variables and initialization ────────────────────────────────────
    BATCH_SIZE = 1
    cap, _, _, _  = read_video(url_video)

    court_model = YOLO(config.KEYPOINTS_COURT_MODEL, task='pose')
    player_model = YOLO(config.PLAYER_POSE_MODEL, task='pose')
    ball_model = YOLO(config.BALL_MODEL, task='detect')

    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    keypoints_court = KeypointsCourt()

    queue = Queue(maxsize=128)
    producer_thread = Thread(target=video_reader, args=(cap, queue))
    producer_thread.start()
    # ───────────────────────────────────────────────────────

    is_first_frame = True
    video_ended = False
    while not video_ended:

        batch_frames = []
        batch_idx = []
        while len(batch_frames) < BATCH_SIZE:

            (frame_idx, frame) = queue.get()

            if frame_idx == -1 and frame is None:
                video_ended = True
                break
            
            batch_frames.append(frame)
            batch_idx.append(frame_idx)

        if len(batch_frames) == 0:
                break    

        if is_first_frame:

            result_keypoints = make_prediction_batch(court_model, batch_frames[0], conf_grade = 0.25)
            kps_obj = result_keypoints[0].keypoints if isinstance(result_keypoints, list) else result_keypoints.keypoints
            if kps_obj is None or len(kps_obj.xy[0]) < 4:
                continue
                
            kps = kps_obj.xy.cpu().numpy()
            keypoints_court.refine_points(batch_frames[0], kps[0])
            keypoints_court.extract_rest_of_kpoints()
            is_first_frame = False

        while len(batch_frames) > 0 and len(batch_frames) < BATCH_SIZE:
            batch_frames.append(batch_frames[-1])
            batch_idx.append(-2)
        
        print(f"Pasando batch a los modelos de deteccion\n")
        result_players = make_track_batch(player_model, batch_frames)
        result_ball = make_prediction_batch(ball_model, batch_frames,  conf_grade = 0.40)

        for i in range(len(batch_frames)):

            frame_idx = batch_idx[i]
            if frame_idx == -2:
                continue

            res_players = result_players[i]
            res_ball = result_ball[i]
        
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
    output_path = os.path.join(config.RAW_JSON_FOLDER_PATH, f"{video_name}_mt_b1.json")
    with open(output_path, 'w') as f:
        json.dump(detection, f, cls=config.NumpyEncoder, indent=2)

    fin = time.time()   
    total = fin - inicio
    print(f"Tiempo -> {total:.6f} segundos")
    print(f"Extraccion finalizada! Raw Data guardada en {output_path}")
