# Rutas de archivos y modelos
import numpy as np


MODEL_PATH = 'models/yolov8-pose-keypoint-court.pt'
DEFAULT_IMAGE_PATH = 'data/inputs/images/frame_002066.png'

# ─── Configuración de visión y ajuste de keypoints ────────────────────────
HALF_WIN_H    = 120      # Ventana horizontal (ancho) para buscar línea H
HALF_WIN_V    = 80       # Ventana vertical (alto) para buscar línea V
HALF_WIN_THIN = 20       # Ventana estrecha perpendicular a cada línea
MIN_LINE_LEN  = 30
MAX_LINE_GAP  = 15
HOUGH_THRESH  = 25
ANGLE_TOL_DEG = 25

real_points = np.array([
    [0.0, 6.95],      # 0: T Superior (A 6.95m hacia arriba de la red)
    [-5.0, 6.95],     # 1: Cristal Izquierdo (Misma altura que la T, pero 5m a la izquierda)
    [5.0, 6.95],      # 2: Cristal Derecho (Misma altura que la T, pero 5m a la derecha)
    [0.0, 0.0],       # 3: Net (El centro exacto, nuestro Origen)
    [0.0, -6.95]      # 4: T Inferior (A 6.95m hacia abajo de la red)
], dtype=np.float32)

