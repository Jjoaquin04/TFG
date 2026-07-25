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
    frame_idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            queue.put((-1, None))
            break

        queue.put((frame_idx, frame))
        frame_idx += 1

def obtain_court_lines(img, best_contour):
    clean_mask = np.zeros(img.shape[:2],dtype=np.uint8) 
    clean_mask = cv2.drawContours(clean_mask, [best_contour], -1, 255, -1)
    gaussian_mask = cv2.GaussianBlur(clean_mask, (23,23), 0)
    canny_img = cv2.Canny(gaussian_mask, 20, 70)
        
    lines = cv2.HoughLines(canny_img, 1, np.pi / 180, threshold=100)
    best_left = None
    best_right = None
    best_top = None
    best_bottom = None
    
    if lines is not None:
        # Calculamos el centro de masa a traves del momentum del contorno
        M = cv2.moments(best_contour)
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else img.shape[0] // 2
        
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            
            #Clasificación geométrica
            if 10 < angle_deg < 80:
                # Pared Izquierda
                if best_left is None: best_left = (rho, theta)
            elif 100 < angle_deg < 170:
                #Pared Derecha
                if best_right is None: best_right = (rho, theta)
            elif 80 <= angle_deg <= 100:
                #Fondo o Red/Inferior
                #Obtenemos el punto de corte de la recta infinita con el borde izq de la pantalla
                y_intercept = rho / np.sin(theta)
                if y_intercept < cy:
                    if best_top is None: best_top = (rho, theta)
                else:
                    if best_bottom is None: best_bottom = (rho, theta)

    return [best_left, best_right, best_top, best_bottom]

def draw_edges_court_connections(frame, court_points, is_mini_court=False):
    for edge in config.COURT_EDGES:
            pt1 = court_points[edge[0]][0]
            pt2 = court_points[edge[1]][0]
            # Seleccionamos el color del índice 3 (mini court) o índice 2 (real court)
            color = edge[3] if is_mini_court else edge[2]
            
            if not is_mini_court and edge[0] == 11 and edge[1] == 12:
                # No pintar la edge de la red en court real
                continue
                
            thickness = 2 if is_mini_court else 5
            cv2.line(frame, (int(round(pt1[0])), int(round(pt1[1]))), 
                            (int(round(pt2[0])), int(round(pt2[1]))), color, thickness)
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