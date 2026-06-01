from dataclasses import dataclass, field
from typing import List
import pandas as pd

@dataclass
class Player:
    id: int
    bbx: List[float] = None # Bounding box actual: [x1, y1, x2, y2]
    keypoints: List[float] = None # Keypoints actual: [x1, y1, x2, y2, ..., xN, yN]
    real_position: List[float] = None # Posición real en la cancha: [x, y]
    history: List[dict] = field(default_factory=list)

    @property
    def current_position(self) -> List[float]:
        return self.real_position
    
    def update(self, frame_idx: int, new_bbx: List[float], new_keypoints: List[float], keypoints_norm: List[float], new_real_position: List[float]):
        self.bbx = new_bbx
        self.keypoints = new_keypoints
        norm_keypoints = self.normalice_keypoints(new_keypoints=new_keypoints)
        self.real_position = new_real_position
        center_x = new_bbx[0] + new_bbx[2] / 2
        center_y = new_bbx[1] + new_bbx[3] / 2

        self.history.append({
            'frame': frame_idx,
            'player_id': self.id,
            'x_min': new_bbx[0],
            'y_min': new_bbx[1],
            'x_max': new_bbx[2],
            'y_max': new_bbx[3],
            'center_x' : center_x,
            'center_y' : center_y,
            'real_x': new_real_position[0],
            'real_y': new_real_position[1],
            'keypoints': new_keypoints,
            'norm_keypoints': norm_keypoints
        })
    
    def normalice_keypoints(self, new_keypoints: List[float]):
        center_hips_x = (new_keypoints[11][0] + new_keypoints[12][0]) / 2.0
        center_hips_y = (new_keypoints[11][1] + new_keypoints[12][1]) / 2.0
        norm_keypoints = []
        for kp in new_keypoints:
            norm_x = kp[0] - center_hips_x
            norm_y = kp[1] - center_hips_y 
            norm_keypoints.append([norm_x,norm_y])

        return norm_keypoints
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
