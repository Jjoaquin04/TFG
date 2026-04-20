import cv2
from ultralytics import YOLO
import config
from core.ball.ball_tracker import BallTracker
from core.court.keypoints import KeypointsCourt
from core.court.mini_court.mini_court import MiniCourt
from core.player.player_tracker import PlayerTracker
from utils import read_video, make_prediction, close_window
from utils.image_video.ImageVideoHandler import draw_bounding_boxes, draw_edges_court_connections, open_window, read_video
from utils.model.InferenceEngine import make_track

# ──────── Variables and initialization ────────────────────────────────────
cap, frame_height , frame_width, fps = read_video(config.VIDEO_PATH)
out = cv2.VideoWriter('data/outputs/output_video.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'),fps=fps, frameSize=(int(round(frame_width)), int(round(frame_height))))

court_model = YOLO(config.KEYPOINTS_COURT_MODEL)
player_model = YOLO(config.PLAYER_POSE_MODEL)
ball_model = YOLO(config.BALL_MODEL)

player_tracker = PlayerTracker(homography_matrix=None)
ball_tracker = BallTracker(homography_matrix=None)

mini_court = MiniCourt(frame_width)
keypoints_court = KeypointsCourt()

first_frame = True
# ───────────────────────────────────────────────────────


while cap.isOpened():

    ret, img = cap.read()

    print(f"Processing frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} / {cap.get(cv2.CAP_PROP_FRAME_COUNT)}", end="\r")

    if not ret:
        break
    
    if first_frame:
        #Prediction and refine keypoints court
        result_keypoints = make_prediction(court_model, img)
        kps = result_keypoints[0].keypoints.xy.cpu().numpy()
        keypoints_court.refine_points(img, kps[0])
        keypoints_court.extract_rest_of_kpoints()

        player_tracker.homography = keypoints_court.H
        first_frame = False

    #Draw court lines
    draw_edges_court_connections(img, keypoints_court.keypoints.reshape(-1, 1, 2))

    result_players = make_track(player_model, img)
    result_ball = make_prediction(ball_model, img)

    if result_players.boxes is not None:
        boxes = result_players.boxes.xyxy.cpu().numpy() 
        track_ids = result_players.boxes.id.cpu().numpy().astype(int) if result_players.boxes.id is not None else None
        keypoints = result_players.keypoints.data.cpu().numpy() if result_players.keypoints is not None else None

    ball_tracker.update(result_ball.boxes.xyxy.cpu().numpy())
    player_tracker.update(track_ids, boxes, keypoints)
    draw_bounding_boxes(img,player_tracker.players.values(), player_tracker.players.keys())  
    image_with_minicourt = mini_court.draw_court(img, player_tracker.get_players_positions())
    out.write(image_with_minicourt)
    
cap.release()   
out.release()

