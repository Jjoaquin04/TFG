import numpy as np

import config
from core.court.keypoints import KeypointsCourt
from utils import read_image, make_prediction, open_window, close_window
import cv2

def main():
    # ─── Carga imagen y modelo ────────────────────────────────────────────────────
    img, _ ,_ = read_image(config.DEFAULT_IMAGE_PATH)

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

    H, mascara_inliners = cv2.findHomography(keypoints_court.keypoints, config.real_points, cv2.RANSAC)
    print("Matriz de homografía H:", H)
    esqueleto_completo_metros = np.array([
        # --- 1. LAS 4 ESQUINAS DEL FONDO ---
        [-5.1, 10.0],   # Esquina Superior Izquierda
        [5.1, 10.0],    # Esquina Superior Derecha
        [-4.9, -10.0],  # Esquina Inferior Izquierda
        [4.9, -10.0],   # Esquina Inferior Derecha
        
        # --- 2. LA RED ---
        [-5.0, 0.0],    # Poste de red Izquierdo
        [5.0, 0.0],     # Poste de red Derecho
        
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
       
    # Extraemos las esquinas proyectadas y las unimos en orden (Superior Izq -> Superior Der -> Inferior Der -> Inferior Izq)
    # Índices en esqueleto_transformado: 0 (Sup Izq), 1 (Sup Der), 2 (Inf Izq), 3 (Inf Der)
    if esqueleto_transformado is not None and len(esqueleto_transformado) >= 4:
        esquinas = [
            esqueleto_transformado[0][0], # Superior Izquierda
            esqueleto_transformado[1][0], # Superior Derecha
            esqueleto_transformado[3][0], # Inferior Derecha
            esqueleto_transformado[2][0]  # Inferior Izquierda
        ]
        pts = np.array([[int(round(x)), int(round(y))] for x, y in esquinas], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(204, 255, 0), thickness=2)
    
    open_window("Keypoint refinados", img)
    close_window()

if __name__ == "__main__":
    main()
