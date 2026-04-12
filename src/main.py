import cv2
import config
from core.court.keypoints import KeypointsCourt
from core.court.mini_court.mini_court import MiniCourt
from utils import read_video, make_prediction, open_window, close_window
from utils.image_video.ImageVideoHandler import draw_edges_court_connections, read_video

# ─── Carga imagen y modelo ────────────────────────────────────────────────────
cap, frame_height , frame_width, fps = read_video(config.VIDEO_PATH)
out = cv2.VideoWriter('data/outputs/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

first_frame = True
mini_court = MiniCourt(frame_width)
keypoints_court = KeypointsCourt()

while cap.isOpened():

    ret, img = cap.read()

    if not ret:
        break
    
    if first_frame:
        result = make_prediction(config.KEYPOINTS_COURT_MODEL, img)
        kps = result[0].keypoints.xy.cpu().numpy()
        keypoints_court.refine_points(img, kps[0])
        keypoints_court.extract_rest_of_kpoints()
        first_frame = False

    draw_edges_court_connections(img, keypoints_court.keypoints.reshape(-1, 1, 2))
    image_with_minicourt = mini_court.draw_court(img)
    open_window("PadelAnalytics", image_with_minicourt)
    out.write(image_with_minicourt)




cap.release()   
out.release()
close_window()

