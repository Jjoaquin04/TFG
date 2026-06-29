import json
import os

import cv2
import numpy as np
import config
from ultralytics import YOLO
from core import BallTracker ,KeypointsCourt, PlayerTracker
from utils import read_video, make_prediction, make_track

def extract(url_video):
    
    # ──────── Variables and initialization ────────────────────────────────────
    cap, _, _, _  = read_video(url_video)

    court_model = YOLO(config.KEYPOINTS_COURT_MODEL, task='pose')
    player_model = YOLO(config.PLAYER_POSE_MODEL, task='pose')
    ball_model = YOLO(config.BALL_MODEL, task='detect')

    player_tracker = PlayerTracker(homography_matrix=None)
    ball_tracker = BallTracker(homography_matrix=None)

    keypoints_court = KeypointsCourt()

    first_frame = True
    # ───────────────────────────────────────────────────────
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while cap.isOpened():

        ret, img = cap.read()
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        print(f"Frame {frame_idx}/{total_frame}\n")
        if not ret:
            break
        
        if first_frame:
            result_keypoints = make_prediction(court_model, img, conf_grade = 0.25)
            kps_obj = result_keypoints[0].keypoints if isinstance(result_keypoints, list) else result_keypoints.keypoints
            if kps_obj is None or len(kps_obj.xy[0]) < 4:
                continue
                
            kps = kps_obj.xy.cpu().numpy()
            keypoints_court.refine_points(img, kps[0])
            keypoints_court.extract_rest_of_kpoints()
            player_tracker.homography = keypoints_court.H
            ball_tracker.homography = keypoints_court.H
            first_frame = False

        result_players = make_track(player_model, img)
        result_ball = make_prediction(ball_model, img,  conf_grade = 0.40)

        boxes, track_ids, keypoints = [], [], None
        if result_players.boxes is not None and len(result_players.boxes) > 0 and result_players.boxes.id is not None:
            boxes = result_players.boxes.xyxy.cpu().numpy() 
            track_ids = result_players.boxes.id.cpu().numpy().astype(int)
            keypoints = result_players.keypoints.data.cpu().numpy() if hasattr(result_players, 'keypoints') and result_players.keypoints is not None else None
            keypoints_norm = result_players.keypoints.xyn.cpu().numpy()

        player_tracker.update(track_ids, boxes, keypoints, keypoints_norm, frame_idx)
        
        if hasattr(result_ball, 'boxes') and result_ball.boxes is not None:
            ball_boxes = result_ball.boxes.xyxy.cpu().numpy()
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
    output_path = os.path.join(config.RAW_JSON_FOLDER_PATH, f"{video_name}.json")
    with open(output_path, 'w') as f:
        json.dump(detection, f, cls=config.NumpyEncoder, indent=2)

    print(f"Extraction finished! Saved raw data to {output_path}")
    




