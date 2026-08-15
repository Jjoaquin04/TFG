import json
import os
import cv2
import time
import numpy as np
import config
import base64
from queue import Queue
from threading import Thread
from tqdm import tqdm
from ultralytics import YOLO
from core import BallTracker ,KeypointsCourt, PlayerTracker
from utils import read_video, make_prediction_batch, make_track_batch, video_reader

def extract(url_video):
    
    inicio = time.time()
    # ──────── Variables and initialization ────────────────────────────────────
    BATCH_SIZE = 16
    cap, _, _, _  = read_video(url_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    player_model = YOLO(config.PLAYER_POSE_MODEL, task='pose')
    ball_model = YOLO(config.BALL_MODEL, task='detect')

    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()

    keypoints_court = KeypointsCourt()

    queue = Queue(maxsize=150)
    frames_thread = Thread(target=video_reader, args=(cap, queue))
    frames_thread.start()
    # ───────────────────────────────────────────────────────

    is_first_frame = True
    video_ended = False
    
    print("Iniciando extracción de datos (jugadores y pelota)...")
    pbar = tqdm(total=total_frames, desc="Extrayendo Frames")
    
    while not video_ended:

        batch_frames = []
        batch_idx = []
        batch_fgmasks = []

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
            first_frame = batch_frames[0]
            keypoints_court.get_delimited_court(first_frame)
            keypoints_court.extract_homography()
            
            _, buffer = cv2.imencode('.jpg', first_frame)
            first_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            is_first_frame = False

        while len(batch_frames) > 0 and len(batch_frames) < BATCH_SIZE:
            batch_frames.append(batch_frames[-1])
            batch_idx.append(-2)
        
        result_players = make_track_batch(player_model, batch_frames, batch_size=BATCH_SIZE)
        result_ball = make_prediction_batch(ball_model, batch_frames, conf_grade=0.2, batch_size=BATCH_SIZE)

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

            player_tracker.update(batch_frames[i], ids, boxes, keypoints, frame_idx)
        
            if hasattr(res_ball, 'boxes') and res_ball.boxes is not None:
                ball_boxes = res_ball.boxes.xyxy.cpu().numpy()
                ball_tracker.update(ball_boxes, frame_idx) 
            else:
                ball_tracker.update([], frame_idx)
        
        pbar.update(len(batch_frames))
        
    pbar.close()
    cap.release()
    ball_history = ball_tracker.get_ball_history()
    players_history = player_tracker.get_players_history()
    court_information = keypoints_court.get_court_information()

    detection = {
        'ball': ball_history,
        'players': players_history,
        'court': court_information,
        'first_frame': first_frame_b64
    }   

    os.makedirs(config.RAW_JSON_FOLDER_PATH, exist_ok=True)
    video_name = os.path.basename(url_video).split('.')[0]
    output_path = os.path.join(config.RAW_JSON_FOLDER_PATH, f"{video_name}.json")
    with open(output_path, 'w') as f:
        json.dump(detection, f, cls=config.NumpyEncoder, indent=2)

    fin = time.time()   
    total = fin - inicio
    print(f"Tiempo -> {total:.6f} segundos")
    print(f"Extraccion finalizada! Raw Data guardada en {output_path}")
    




