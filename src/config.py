# Rutas de archivos y modelos
import numpy as np


MODEL_PATH = 'models/yolov8-pose-keypoint-court.pt'
DEFAULT_IMAGE_PATH = 'data/inputs/images/frame_000011.png'

# ─── Configuración de visión y ajuste de keypoints ────────────────────────
HALF_WIN_H    = 120      # Ventana horizontal (ancho) para buscar línea H
HALF_WIN_V    = 80       # Ventana vertical (alto) para buscar línea V
HALF_WIN_THIN = 20       # Ventana estrecha perpendicular a cada línea
MIN_LINE_LEN  = 30
MAX_LINE_GAP  = 15
HOUGH_THRESH  = 25
ANGLE_TOL_DEG = 25

real_points_model = np.array([
    [-5.0, 6.95],     # 1: Cristal Izquierdo Superior
    [0.0, 6.95],      # 0: T Superior    
    [5.0, 6.95],      # 2: Cristal Derecho Superior
    [0.0, 0.0],       # 3: Net
    [-5.0, -6.95],    # 4: Cristal Izquierdo Inferior
    [0.0, -6.95],     # 5: T Inferior      
    [5.0, -6.95]      # 6: Cristal Derecho Inferior
], dtype=np.float32)

rest_real_points_model = np.array([
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

points_court = np.concatenate((real_points_model.reshape(-1, 1, 2), rest_real_points_model), axis=0)

COURT_EDGES = [
    # Formato: (Indice1, Indice2, Color_Real_Court (BGR), Color_Mini_Court (BGR))
    # 1. Contorno de la pista (Esquinas)
    (7, 8, (0, 255, 0), (54, 69, 79)),   # Fondo Superior: Verde Neón en Real, Gris Fuerte en Mini
    (9, 10, (0, 255, 0), (54, 69, 79)),  # Fondo Inferior
    (7, 9, (0, 255, 0), (54, 69, 79)),   # Lateral Izquierdo
    (8, 10, (0, 255, 0), (54, 69, 79)),  # Lateral Derecho
    
    # 2. Líneas de Servicio (Intersecciones de Cristal/T)
    (0, 2, (0, 255, 0), (54, 69, 79)),   # Línea de servicio Superior (pasa por T Sup)
    (4, 6, (0, 255, 0), (54, 69, 79)),   # Línea de servicio Inferior (pasa por T Inf)
    
    # 3. Línea Central de Servicio (une las dos T)
    (1, 5, (0, 255, 0), (54, 69, 79)),   # De T Superior a T Inferior
    
    # 4. La Red
    (11, 12, (0, 0, 255), (80, 80, 220)),  # Poste Izquierdo a Poste Derecho: Rojo vibrante Real, Rojo suave en Mini
]