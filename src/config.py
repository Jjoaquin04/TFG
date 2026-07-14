# Rutas de archivos, modelos, estructuras y clase cte
import json
import numpy as np

USE_OPEN_VINO = True

if USE_OPEN_VINO:
    KEYPOINTS_COURT_MODEL = 'models/yolov8-court-keypoint_openvino_model'
    PLAYER_POSE_MODEL = 'models/yolov8-player-pose_openvino_model'
    BALL_MODEL = 'models/yolov8-ball-bbx_openvino_model'
else:
    KEYPOINTS_COURT_MODEL = 'models/yolov8-court-keypoint.pt'
    PLAYER_POSE_MODEL = 'models/yolov8-player-pose.pt'
    BALL_MODEL = 'models/yolov8-ball-bbx.pt'

RAW_JSON_FOLDER_PATH = 'data/outputs/json/raw_json'
INTERP_JSON_PATH = 'data/outputs/json'

# ─── Configuración de visión y ajuste de keypoints ────────────────────────
HALF_WIN_H    = 120      # Ventana horizontal (ancho) para buscar línea H
HALF_WIN_V    = 80       # Ventana vertical (alto) para buscar línea V
HALF_WIN_THIN = 20       # Ventana estrecha perpendicular a cada línea
MIN_LINE_LEN  = 30
MAX_LINE_GAP  = 15
HOUGH_THRESH  = 25
ANGLE_TOL_DEG = 25

real_points= np.array([
    [-5.0, 10.0],   # Esquina Superior Izquierda
    [5.0, 10.0],    # Esquina Superior Derecha
    [-5.0, -10.0],  # Esquina Inferior Izquierda
    [5.0, -10.0],   # Esquina Inferior Derecha
    [0.0, 6.95],    # T Superior    
    [0.0, -6.95],   # T Inferior      
], dtype=np.float32)

rest_real_points = np.array([
    [-5.0, 6.95],   # Cristal Izquierdo Superior
    [5.0, 6.95],    # Cristal Derecho Superior
    [-5.0, -6.95],  # Cristal Izquierdo Inferior
    [5.0, -6.95],    # Cristal Derecho Inferior
    [0.0, 0.0],     # Net
    [-5.0, 0.0],    # Poste de red Izquierdo
    [5.0, 0.0],     # Poste de red Derecho
    # --- 4. TRANSICIÓN REJA / CRISTAL ---
    [-5.0, 6.0],    # Empieza reja Superior Izquierda
    [5.0, 6.0],     # Empieza reja Superior Derecha
    [-5.0, -6.0],   # Empieza reja Inferior Izquierda
    [5.0, -6.0]     # Empieza reja Inferior Derecha

], dtype=np.float32).reshape(-1, 1, 2)

points_court = np.concatenate((real_points.reshape(-1, 1, 2), rest_real_points), axis=0)

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
class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

types_of_mask = {
    "mask1" : [[127, 99, 97], [133, 180, 255]], # Harcodeado a traves del metodo tune_mask_live.py 
    "mask2" : [[93, 153, 172], [123, 253, 255]], 
    "mask3" : [[98, 158, 205], [128, 255, 255]], 
    "mask4" : [[97, 172, 205],[127, 255, 255]]
}