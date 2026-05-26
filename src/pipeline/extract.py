import cv2
import numpy as np
import config
from ultralytics import YOLO
from core import BallTracker ,KeypointsCourt, PlayerTracker
from utils import read_video, make_prediction, make_track

def extract():
    
    # ──────── Variables and initialization ────────────────────────────────────
    cap, _, _, _  = read_video(config.VIDEO_PATH)

    court_model = YOLO(config.KEYPOINTS_COURT_MODEL)
    player_model = YOLO(config.PLAYER_POSE_MODEL)
    ball_model = YOLO(config.BALL_MODEL)

    player_tracker = PlayerTracker(homography_matrix=None)
    ball_tracker = BallTracker(homography_matrix=None)

    keypoints_court = KeypointsCourt()

    first_frame = True
    # ───────────────────────────────────────────────────────

    while cap.isOpened():

        ret, img = cap.read()
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if not ret:
            break
        
        if first_frame:
            #Prediction and refine keypoints court
            result_keypoints = make_prediction(court_model, img)
            kps = result_keypoints[0].keypoints.xy.cpu().numpy()
            keypoints_court.refine_points(img, kps[0])
            keypoints_court.extract_rest_of_kpoints()

            player_tracker.homography = keypoints_court.H
            ball_tracker.homography = keypoints_court.H
            first_frame = False

        result_players = make_track(player_model, img)
        result_ball = make_prediction(ball_model, img)

        boxes, track_ids, keypoints = [], [], None
        if result_players.boxes is not None and len(result_players.boxes) > 0:
            boxes = result_players.boxes.xyxy.cpu().numpy() 
            track_ids = result_players.boxes.id.cpu().numpy().astype(int) if result_players.boxes.id is not None else None
            keypoints = result_players.keypoints.data.cpu().numpy() if result_players.keypoints is not None else None

        player_tracker.update(track_ids, boxes, keypoints, frame_idx)
        
        if hasattr(result_ball, 'boxes') and result_ball.boxes is not None:
            ball_boxes = result_ball.boxes.xyxy.cpu().numpy()
            if len(ball_boxes) > 0:
                ball_tracker.update(ball_boxes[0], frame_idx) 
            else:
                ball_tracker.update(None, frame_idx)
        else:
            ball_tracker.update(None, frame_idx)
        
    cap.release()   



