import cv2
import numpy as np
import config
from utils.image_video.ImageVideoHandler import get_roi_clamped, preprocess_roi, get_segments, classify_segments, fit_line_svd

class KeypointsCourt:
    def __init__(self):
        self.keypoints = np.array([], dtype=np.float32).reshape(0, 2)
    
    def extract_rest_of_kpoints(self):
        self.extract_homography()
        rest_points = cv2.perspectiveTransform(config.rest_real_points_model, self.inverse_H)
        self.keypoints = np.vstack([self.keypoints, rest_points.reshape(-1, 2)])
    
    def refine_points(self, img, kps):
        """
        Itera sobre los keypoints y los afina llamando al método numérico interno.
        """
        for kp in kps:
            cx, cy = kp[0], kp[1]
            exact_x, exact_y = self._refine_single_point(img, cx, cy)
            self.keypoints = np.vstack([self.keypoints, [exact_x, exact_y]])

    def extract_homography(self):
        self.H, _ = cv2.findHomography(self.keypoints, config.real_points_model, cv2.RANSAC)
        self.inverse_H = np.linalg.inv(self.H)
        
    def _refine_single_point(self, img, cx, cy):
        roi_H, ox_H, oy_H = get_roi_clamped(img, cx, cy,
                                              half_w=config.HALF_WIN_H,
                                              half_h=config.HALF_WIN_THIN)
        edges_H = preprocess_roi(roi_H, mask_color=True)

        segs_H_raw = get_segments(edges_H, min_len=config.MIN_LINE_LEN // 2)
        h_segs = classify_segments(segs_H_raw, 'H')

        if len(h_segs) == 0:
            edges_H_nm = preprocess_roi(roi_H, mask_color=False)
            segs_H_raw = get_segments(edges_H_nm, min_len=config.MIN_LINE_LEN // 2)
            h_segs = classify_segments(segs_H_raw, 'H')

        roi_V, ox_V, oy_V = get_roi_clamped(img, cx, cy,
                                              half_w=config.HALF_WIN_THIN,
                                              half_h=config.HALF_WIN_V)
        edges_V = preprocess_roi(roi_V, mask_color=True)

        segs_V_raw = get_segments(edges_V, min_len=config.MIN_LINE_LEN // 2)
        v_segs = classify_segments(segs_V_raw, 'V')

        if len(v_segs) == 0:
            edges_V_nm = preprocess_roi(roi_V, mask_color=False)
            segs_V_raw = get_segments(edges_V_nm, min_len=config.MIN_LINE_LEN // 2)
            v_segs = classify_segments(segs_V_raw, 'V')

        if len(h_segs) == 0 or len(v_segs) == 0:
            return cx, cy

        h_segs_global = [[xa + ox_H, ya + oy_H, xb + ox_H, yb + oy_H] for xa, ya, xb, yb in h_segs]
        v_segs_global = [[xa + ox_V, ya + oy_V, xb + ox_V, yb + oy_V] for xa, ya, xb, yb in v_segs]

        a1, b1, c1 = fit_line_svd(h_segs_global)
        a2, b2, c2 = fit_line_svd(v_segs_global)

        try:
            A = np.array([[a1, b1], [a2, b2]])
            C = np.array([c1, c2])
            sol = np.linalg.solve(A, C)
            return sol[0], sol[1]
        except np.linalg.LinAlgError:
            return cx, cy


