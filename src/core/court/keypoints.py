import cv2
import numpy as np
import math
import config
from utils import obtain_court_lines 

class KeypointsCourt:
    def __init__(self):
        self.keypoints = np.array([], dtype=np.float32).reshape(0, 2)
    
    def extract_rest_of_kpoints(self):
        self.extract_homography()
        rest_points = cv2.perspectiveTransform(config.rest_real_points, self.inverse_H)
        self.keypoints = np.vstack([self.keypoints, rest_points.reshape(-1, 2)])
    
    def refine_points(self, img, kps):
        self.get_delimited_court(img)
        index_to_refine = [1, 5]
        best_mask_img_inverted = cv2.bitwise_not(self.best_mask_img)
        
        refined_kps = kps.copy()
        for idx in index_to_refine:
            pt = refined_kps[idx].flatten()
            x_center, y_center = int(pt[0]), int(pt[1])
            
            window = 40
            y1 = max(0, y_center - window)
            y2 = min(img.shape[0], y_center + window)
            x1 = max(0, x_center - window)
            x2 = min(img.shape[1], x_center + window)
            
            mask_window = best_mask_img_inverted[y1:y2, x1:x2].copy()
            if mask_window.size == 0: continue

            kernel = np.ones((7,7), np.uint8)
            mask_window = cv2.morphologyEx(mask_window, cv2.MORPH_OPEN, kernel)

            dst = cv2.cornerHarris(np.float32(mask_window), blockSize=7, ksize=5, k=0.04)
            _, max_val, _, max_loc = cv2.minMaxLoc(dst)
            cx, cy = max_loc
            refined_kps[idx, 0] = x1 + cx
            refined_kps[idx, 1] = y1 + cy

            self.keypoints = np.vstack([self.keypoints, np.array(refined_kps[idx]).reshape(-1,2)])

        #Extraer el resto de los puntos
        self.extract_rest_of_kpoints()


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
                
            #Elegir el de mayor área
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            #Filtrar manchas pequeñas 
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
        self.H, _ = cv2.findHomography(self.keypoints, config.real_points, cv2.RANSAC)
        self.inverse_H = np.linalg.inv(self.H)

    def get_court_information(self):
        return [self.keypoints, self.H]
