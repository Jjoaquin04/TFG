import numpy as np

import config
from core.court.keypoints import KeypointsCourt
from core.court.mini_court.mini_court import MiniCourt
from utils import read_image, make_prediction, open_window, close_window
import cv2

from utils.image_video.ImageVideoHandler import draw_edges_court_connections

def main():
    # ─── Carga imagen y modelo ────────────────────────────────────────────────────
    img, _ , frame_width = read_image(config.DEFAULT_IMAGE_PATH)

    result = make_prediction(config.MODEL_PATH, img)

    # Extraemos los keypoints del modelo
    kps = result[0].keypoints.xy.cpu().numpy()

    # ─── Creamos el objeto KeypointsCourt y refinamos los puntos ─────────────────
    keypoints_court = KeypointsCourt()
    keypoints_court.refine_points(img, kps[0])  # kps[0] ya que yolo te devuelve shape (1, num_kps, 2)
    print("Keypoints refinados (coordenadas de imagen):", keypoints_court.keypoints)
    img_kps = img.copy()
    for i, (x, y) in enumerate(keypoints_court.keypoints):
        print(f"Keypoint {i}: ({x:.2f}, {y:.2f})")
        cv2.circle(img_kps, (int(round(x)), int(round(y))), radius=5, color=(0, 255, 0), thickness=-1)
    open_window("Keypoints refinados", img_kps) 

    H, _ = cv2.findHomography(keypoints_court.keypoints, config.real_points_model, cv2.RANSAC)
    H_inversa = np.linalg.inv(H)

    esqueleto_transformado = cv2.perspectiveTransform(config.rest_real_points_model, H_inversa)
    keypoints_court.append_list_of_points(esqueleto_transformado.reshape(-1, 2))

    # Dibujamos los keypoints refinados + esqueleto completo
    for i, (ex, ey) in enumerate(keypoints_court.keypoints):
       cv2.circle(img, (int(round(ex)),int(round(ey))),radius=5,color=(0,0,255),thickness=-1)
       
    draw_edges_court_connections(img, keypoints_court.keypoints.reshape(-1, 1, 2))
    
    mini_court = MiniCourt(frame_width)
    image_with_minicourt = mini_court.draw_court(img)

    open_window("Keypoints refinados + Esqueleto proyectado + MiniCourt", image_with_minicourt)
    close_window()


if __name__ == "__main__":
    main()
