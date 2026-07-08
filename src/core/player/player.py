from dataclasses import dataclass, field
from typing import List
import pandas as pd
import cv2
import numpy as np

@dataclass
class Player:
    id: int
    bbx: List[float] = None # Bounding box actual: [x1, y1, x2, y2]
    keypoints: List[float] = None # Keypoints actual: [x1, y1, x2, y2, ..., xN, yN]
    history: dict = field(default_factory=dict)
    footprint: List[float] = None
        
    def update(self, frame, frame_idx: int, new_bbx: List[float], new_keypoints: List[float]):
        self.bbx = new_bbx
        self.keypoints = new_keypoints
        
        if self.footprint is None and frame is not None:
            self.footprint = self._extract_footprint(frame, new_bbx, new_keypoints)

        self.history[frame_idx] = ({
            'frame': frame_idx,
            'player_id': self.id,
            'x_min': new_bbx[0],
            'y_min': new_bbx[1],
            'x_max': new_bbx[2],
            'y_max': new_bbx[3],
            'keypoints': new_keypoints,
            'shirt_color_hsv': self.footprint
        })
        
    def _extract_footprint(self, frame, bbx, keypoints):

        try:
            if keypoints and len(keypoints) > 13:
                left_shoulder = keypoints[5][:2]
                right_soulder = keypoints[6][:2]
                left_hip = keypoints[11][:2]
                right_hip = keypoints[12][:2]

                if all(pt[0] > 0.0 and pt[1] > 0 for pt in [left_shoulder, right_soulder, left_hip, right_hip]):
                    pts = np.array([left_shoulder, right_soulder, left_hip, right_hip],dtype=np.int32 )
                    pts = pts.reshape(-1,1,2)

                    #Mascara negra del tamaño del frame
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)

                    x, y, w, h = cv2.boundingRect(pts)
                    frame_crop = frame[y:y+h, x:x+w]
                    mask_crop = mask[y:y+h, x:x+w]

                    hsv_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
                    valid_pixels = hsv_crop[mask_crop == 255]

                    if len(valid_pixels) > 0:
                        median_h = float(np.median(valid_pixels[:, 0]))
                        median_s = float(np.median(valid_pixels[:, 1]))
                        median_v = float(np.median(valid_pixels[:, 2]))
                        return [median_h, median_s, median_v]
                
            x1, y1, x2, y2 = map(int, bbx)
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            bbx_w = x2 - x1
            bbx_h = y2 - y1
    
            #Coger solo el centro-pecho
            crop_x1 = int(x1 + bbx_w * 0.2)
            crop_x2 = int(x1 + bbx_w * 0.8)
            crop_y1 = int(y1 + bbx_h * 0.1)
            crop_y2 = int(y1 + bbx_h * 0.4)

            shirt_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if shirt_crop.size == 0:
                return [0.0, 0.0, 0.0]
                
            hsv_crop = cv2.cvtColor(shirt_crop, cv2.COLOR_BGR2HSV)
            median_h = float(np.median(hsv_crop[:, :, 0]))
            median_s = float(np.median(hsv_crop[:, :, 1]))
            median_v = float(np.median(hsv_crop[:, :, 2]))
            
            return [median_h, median_s, median_v]
        except:
            return [0.0,0.0,0.0]
    
    
