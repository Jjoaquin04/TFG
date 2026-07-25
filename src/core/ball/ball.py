from dataclasses import dataclass, field
from typing import List
import pandas as pd

@dataclass
class Ball:

    history: List[dict] = field(default_factory=list)

    def update(self, frame_idx: int, ball_detections):
        dets = []
        for bbx in ball_detections:
            dets.append({
                'x_min': float(bbx[0]),
                'y_min': float(bbx[1]),
                'x_max': float(bbx[2]),
                'y_max': float(bbx[3])
            })
    
        self.history.append({
            'frame': frame_idx,
            'detections': dets
        })
    