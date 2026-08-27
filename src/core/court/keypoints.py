import cv2
import numpy as np
import math
import config
from utils import obtain_court_lines, draw_edges_court_connections 

class KeypointsCourt:
    def __init__(self):
        self.keypoints = np.array([], dtype=np.float32).reshape(0, 2)
    
    def extract_rest_of_kpoints(self):
        self.extract_homography()
        rest_points = cv2.perspectiveTransform(config.rest_real_points, self.inverse_H)
        self.keypoints = np.vstack([self.keypoints, rest_points.reshape(-1, 2)])

    def get_delimited_court(self, img):
        self.best_mask_img = None
        self.best_mask = None
        solidity_prev = -math.inf
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        for idx, (mask_name, mask_bounds) in enumerate(config.types_of_mask.items()):
            masked_img = cv2.inRange(img_hsv,  np.array(mask_bounds[0]), np.array(mask_bounds[1]))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            masked_img_closed = cv2.morphologyEx(masked_img, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(masked_img_closed, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
                
            # Elegir el de mayor área
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            # Filtrar manchas pequeñas 
            area = cv2.contourArea(contour)
            if area < 5000:
                continue
                
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidity = float(area)/hull_area
            if solidity > solidity_prev:
                solidity_prev = solidity
                self.best_mask = mask_bounds
                self.best_mask_img = masked_img
                self.best_contour = contour
    
        court_lines = obtain_court_lines(img, self.best_contour)
        tl = self._get_intersection(court_lines[0], court_lines[2])
        tr = self._get_intersection(court_lines[1], court_lines[2])
        bl = self._get_intersection(court_lines[0], court_lines[3])
        br = self._get_intersection(court_lines[1], court_lines[3])
        
        # Alineación horizontal de las esquinas de la moqueta
        if tl is not None and tr is not None:
            best_y_top = min(tl[1], tr[1])
            tl = (tl[0], best_y_top)
            tr = (tr[0], best_y_top)
            
        if bl is not None and br is not None:
            best_y_bottom = max(bl[1], br[1])
            bl = (bl[0], best_y_bottom)
            br = (br[0], best_y_bottom)
        
        self.keypoints = np.vstack([self.keypoints, np.array([tl, tr, bl, br]).reshape(-1, 2)])

    def _get_intersection(self, l1, l2):
        if l1 is None or l2 is None: return None
        A = np.array([
            [np.cos(l1[1]), np.sin(l1[1])],
            [np.cos(l2[1]), np.sin(l2[1])]
        ])
        b = np.array([l1[0], l2[0]])
        try:
            x, y = np.linalg.solve(A, b)
            return int(np.round(x)), int(np.round(y))
        except np.linalg.LinAlgError:
            return None       

    def extract_homography(self):
        src_pts = self.keypoints[:4].copy()
        dst_pts = config.real_points[:4].copy().reshape(-1, 2)
            
        self.H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        self.inverse_H = np.linalg.inv(self.H)

    def adjust_points_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            dists = np.linalg.norm(self.keypoints - np.array([x, y]), axis=1)
            if np.min(dists) < 50:
                self.dragging_point_idx = np.argmin(dists)
        elif event == cv2.EVENT_MOUSEMOVE:
            if getattr(self, 'dragging_point_idx', -1) != -1:
                self.keypoints[self.dragging_point_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point_idx = -1

    def interactive_adjustment(self, frame):
        self.dragging_point_idx = -1
        window_name = 'Ajuste de Pista (Arrastra los puntos, ENTER para continuar)'
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(window_name, self.adjust_points_mouse_callback)
        
        while True:
            display_frame = frame.copy()
            draw_edges_court_connections(display_frame, self.keypoints.reshape(-1, 1, 2))
            for i, pt in enumerate(self.keypoints):
                cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i), (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break
                
        cv2.destroyAllWindows()

        # Recalcular H basándonos en las 4 esquinas de la moqueta (por si se ajustaron)
        self.H, _ = cv2.findHomography(self.keypoints[:4], config.real_points.copy().reshape(-1, 2), cv2.RANSAC)
        self.inverse_H = np.linalg.inv(self.H)
        
        return self.keypoints.copy(), self.H
