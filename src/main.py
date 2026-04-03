import numpy as np

import config
from core.court.keypoints import KeypointsCourt
from utils import read_image, make_prediction, open_window, close_window
import cv2

def main():
    # ─── Carga imagen y modelo ────────────────────────────────────────────────────
    img, _ ,_ = read_image(config.DEFAULT_IMAGE_PATH)

    result = make_prediction(config.MODEL_PATH, img)
    cv2.imshow("Predicción YOLO", result[0].plot())
    
    # Extraemos los keypoints del modelo
    kps = result[0].keypoints.xy.cpu().numpy()

    # ─── Creamos el objeto KeypointsCourt y refinamos los puntos ─────────────────
    keypoints_court = KeypointsCourt()
    keypoints_court.refine_points(img, kps[0])  # kps[0] ya que yolo te devuelve shape (1, num_kps, 2)
    print("Keypoints refinados (coordenadas de imagen):", keypoints_court.keypoints)
    H, mascara_inliners = cv2.findHomography(keypoints_court.keypoints, config.real_points, cv2.RANSAC)

    esqueleto_completo_metros = np.array([
        # --- 1. LAS 4 ESQUINAS DEL FONDO ---
        [-5.0, 10.0],   # Esquina Superior Izquierda
        [5.0, 10.0],    # Esquina Superior Derecha
        [-5.0, -10.0],  # Esquina Inferior Izquierda
        [5.0, -10.0],   # Esquina Inferior Derecha
        
        # --- 2. LA RED ---
        [-5.0, 0.0],    # Poste de red Izquierdo
        [5.0, 0.0],     # Poste de red Derecho
        
        # --- 3. LÍNEAS DE SAQUE (Laterales) ---
        [-5.0, -6.95],  # Choque saque inferior con cristal Izq
        [5.0, -6.95],   # Choque saque inferior con cristal Der
        
        # --- 4. TRANSICIÓN REJA / CRISTAL ---
        [-5.0, 6.0],    # Empieza reja Superior Izquierda
        [5.0, 6.0],     # Empieza reja Superior Derecha
        [-5.0, -6.0],   # Empieza reja Inferior Izquierda
        [5.0, -6.0]     # Empieza reja Inferior Derecha

    ], dtype=np.float32).reshape(-1, 1, 2)

    H_inversa = np.linalg.inv(H)

    esqueleto_transformado = cv2.perspectiveTransform(esqueleto_completo_metros, H_inversa)
    print("Esqueleto transformado a coordenadas de imagen:", esqueleto_transformado)
    keypoints_court.append_list_of_points(esqueleto_transformado.reshape(-1, 2))

    # Dibujamos los keypoints refinados + esqueleto completo
    for i, (ex, ey) in enumerate(keypoints_court.keypoints):
       cv2.circle(img, (int(round(ex)),int(round(ey))),radius=5,color=(0,0,255),thickness=-1)    
    
    open_window("Keypoint refinados", img)
    close_window()


if __name__ == "__main__":
    main()
