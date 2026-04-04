import cv2
import numpy as np
import config
from utils import get_roi_clamped

class KeypointsCourt:
    def __init__(self):
        self.keypoints = np.array([], dtype=np.float32).reshape(0, 2)
    
    def append_list_of_points(self, points):
        self.keypoints = np.vstack([self.keypoints, points])
    
        
    def refine_points(self, img, kps):
        """
        Itera sobre los keypoints y los afina llamando al método numérico interno.
        """
        for kp in kps:
            cx, cy = kp[0], kp[1]
            exact_x, exact_y = self._refine_single_point(img, cx, cy)
            self.keypoints = np.vstack([self.keypoints, [exact_x, exact_y]])

    def _refine_single_point(self, img, cx, cy):
        roi_H, ox_H, oy_H = get_roi_clamped(img, cx, cy,
                                              half_w=config.HALF_WIN_H,
                                              half_h=config.HALF_WIN_THIN)
        edges_H = self._preprocess_roi(roi_H, mask_color=True)

        segs_H_raw = self._get_segments(edges_H, min_len=config.MIN_LINE_LEN // 2)
        h_segs = self._classify_segments(segs_H_raw, 'H')

        if len(h_segs) == 0:
            edges_H_nm = self._preprocess_roi(roi_H, mask_color=False)
            segs_H_raw = self._get_segments(edges_H_nm, min_len=config.MIN_LINE_LEN // 2)
            h_segs = self._classify_segments(segs_H_raw, 'H')

        roi_V, ox_V, oy_V = get_roi_clamped(img, cx, cy,
                                              half_w=config.HALF_WIN_THIN,
                                              half_h=config.HALF_WIN_V)
        edges_V = self._preprocess_roi(roi_V, mask_color=True)

        segs_V_raw = self._get_segments(edges_V, min_len=config.MIN_LINE_LEN // 2)
        v_segs = self._classify_segments(segs_V_raw, 'V')

        if len(v_segs) == 0:
            edges_V_nm = self._preprocess_roi(roi_V, mask_color=False)
            segs_V_raw = self._get_segments(edges_V_nm, min_len=config.MIN_LINE_LEN // 2)
            v_segs = self._classify_segments(segs_V_raw, 'V')

        if len(h_segs) == 0 or len(v_segs) == 0:
            return cx, cy

        h_segs_global = [[xa + ox_H, ya + oy_H, xb + ox_H, yb + oy_H] for xa, ya, xb, yb in h_segs]
        v_segs_global = [[xa + ox_V, ya + oy_V, xb + ox_V, yb + oy_V] for xa, ya, xb, yb in v_segs]

        a1, b1, c1 = self._fit_line_svd(h_segs_global)
        a2, b2, c2 = self._fit_line_svd(v_segs_global)

        try:
            A = np.array([[a1, b1], [a2, b2]])
            C = np.array([c1, c2])
            sol = np.linalg.solve(A, C)
            return sol[0], sol[1]
        except np.linalg.LinAlgError:
            return cx, cy

    def _preprocess_roi(self, roi, mask_color=True):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if mask_color:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_w = cv2.inRange(hsv, np.array([0,   0, 160]), np.array([180, 50, 255]))
            mask_g = cv2.inRange(hsv, np.array([0,   0, 120]), np.array([180, 40, 200]))
            mask = cv2.bitwise_or(mask_w, mask_g)
            mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        blur  = cv2.GaussianBlur(gray, (5, 5), 1)
        return cv2.Canny(blur, 20, 80, apertureSize=3)

    def _get_segments(self, edges, min_len=-1, max_gap=-1, thresh=-1):
        min_len = config.MIN_LINE_LEN if min_len == -1 else min_len
        max_gap = config.MAX_LINE_GAP if max_gap == -1 else max_gap
        thresh = config.HOUGH_THRESH if thresh == -1 else thresh
        segs = cv2.HoughLinesP(edges, 1, np.pi/180, thresh, minLineLength=min_len, maxLineGap=max_gap)
        return segs if segs is not None else np.array([]).reshape(0,1,4)

    def _classify_segments(self, segs, target='H', tol=None):
        tol = config.ANGLE_TOL_DEG if tol is None else tol
        out = []
        for s in segs:
            xa, ya, xb, yb = s[0]
            angle = np.degrees(np.arctan2(abs(yb - ya), abs(xb - xa)))
            if target == 'H' and angle < tol: out.append(s[0])
            elif target == 'V' and angle > (90 - tol): out.append(s[0])
        return out

    def _fit_line_svd(self, segs_list):
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

