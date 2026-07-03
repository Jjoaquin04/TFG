import cv2
import numpy as np
import config

def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    height_img, width_img, fps = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FPS)
    return cap, height_img, width_img, fps

def open_window(window_name,img):
    cv2.namedWindow(window_name,cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(window_name, img)

def close_window():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_reader(cap, queue):
    print("Leyendo video\n")
    while cap.isOpened():
        ret, frame = cap.read()
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"FRAME_IDX ----> {frame_idx}")
        if not ret:
            queue.put((-1, None))
            break

        queue.put((frame_idx, frame))        

def get_roi_clamped(image, x_center, y_center, half_w, half_h):
    # Devuelve la ROI y sus coordenadas de origen (para mapear de vuelta).
    x0 = max(int(x_center) - half_w, 0)
    x1 = min(int(x_center) + half_w, image.shape[1])
    y0 = max(int(y_center) - half_h, 0)
    y1 = min(int(y_center) + half_h, image.shape[0])
    return image[y0:y1, x0:x1].copy(), x0, y0 # ROI + offsets para coordenadas globales

def preprocess_roi(roi, mask_color=True):
    # Preprocesa la ROI: máscara opcional + Canny.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if mask_color:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Blanco puro de líneas de pista
        mask_w = cv2.inRange(hsv, np.array([0,   0, 160]),
                                  np.array([180, 50, 255]))
        # Líneas que pueden verse grises por perspectiva/luz
        mask_g = cv2.inRange(hsv, np.array([0,   0, 120]),
                                  np.array([180, 40, 200]))
        mask = cv2.bitwise_or(mask_w, mask_g)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    blur  = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 20, 80, apertureSize=3)
    return edges

def get_segments(edges, min_len=config.MIN_LINE_LEN,
                 max_gap=config.MAX_LINE_GAP, thresh=config.HOUGH_THRESH):
    segs = cv2.HoughLinesP(edges, 1, np.pi/180,
                           thresh, minLineLength=min_len,
                           maxLineGap=max_gap)
    return segs if segs is not None else np.array([]).reshape(0,1,4)

def classify_segments(segs, target='H', tol=config.ANGLE_TOL_DEG):
    # Filtra segmentos por orientación (H o V).
    out = []
    for s in segs:
        xa, ya, xb, yb = s[0]
        angle = np.degrees(np.arctan2(abs(yb - ya), abs(xb - xa)))
        if target == 'H' and angle < tol:
            out.append(s[0])
        elif target == 'V' and angle > (90 - tol):
            out.append(s[0])
    return out

def fit_line_svd(segs_list):
    # Ajuste SVD sobre todos los puntos de los segmentos → (a, b, c): ax+by=c.
    pts = []
    for xa, ya, xb, yb in segs_list:
        pts.extend([(xa, ya), (xb, yb)])
    pts = np.array(pts, dtype=np.float64)
    cx_f, cy_f = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - [cx_f, cy_f])
    dx, dy = Vt[0]
    a, b = -dy, dx
    c = a * cx_f + b * cy_f
    return a, b, c

def draw_edges_court_connections(frame, court_points, is_mini_court=False):
    for edge in config.COURT_EDGES:
            pt1 = court_points[edge[0]][0]
            pt2 = court_points[edge[1]][0]
            # Seleccionamos el color del índice 3 (mini court) o índice 2 (real court)
            color = edge[3] if is_mini_court else edge[2]
            
            if not is_mini_court and edge[0] == 11 and edge[1] == 12:
                # No pintar la edge de la red en court real
                continue
                
            cv2.line(frame, (int(round(pt1[0])), int(round(pt1[1]))), 
                            (int(round(pt2[0])), int(round(pt2[1]))), color, 2)
    return frame

def draw_bounding_boxes(frame, bbx, ids=None):
    list_ids = list(ids) if ids is not None else [None] * len(bbx)
    for i, obj in enumerate(bbx):
        x1, y1, x2, y2 = int(round(obj[0])), int(round(obj[1])), int(round(obj[2])), int(round(obj[3]))
        
        id_val = int(list_ids[i]) if ids is not None else 0
        color = (0, 255, 255) if ids is None else (int(id_val * 50 % 256), int(255 - (id_val * 50 % 256)), 150)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        if ids is not None:
            cv2.putText(frame, f'ID: {id_val}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def draw_comet_tail(frame, pt1, pt2, color, num_points):

    bg_color = (220, 220, 220) # Bg color aproximado
    pts_x = np.linspace(pt1[0], pt2[0], num_points)
    pts_y = np.linspace(pt1[1], pt2[1], num_points)
    
    for i in range(num_points):
        ratio = i / (num_points - 1)
        # como crede el radio
        radius = int(2 + 2 * ratio)
        
        # Interpolar efecto de desvanecimiento hacia la cola
        b = int(bg_color[0] * (1 - ratio) + color[0] * ratio)
        g = int(bg_color[1] * (1 - ratio) + color[1] * ratio)
        r = int(bg_color[2] * (1 - ratio) + color[2] * ratio)
        cv2.circle(frame, (int(pts_x[i]), int(pts_y[i])), radius, (b, g, r), -1)
    
    cv2.circle(frame, pt2, 7, (255, 255, 255), 1)
    
    return frame