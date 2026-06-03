from dataclasses import dataclass, field
from typing import List
import pandas as pd

@dataclass
class Ball:

    history: List[dict] = field(default_factory=list)
    bbx: List[float] = None # Bounding box actual: [x1, y1, x2, y2]
    real_position: List[float] = None # Posición real en la cancha: [x, y]

    def get_bbx(self):
        return self.bbx

    def get_real_position(self):
        return self.real_position
    
    def update(self,frame_idx: int,  bbx: List[float], real_position: tuple[float, float]):
        self.bbx = bbx
        self.real_position = real_position,
        self.history.append({
            'frame': frame_idx,
            'x_min': bbx[0] if bbx is not None else None,
            'y_min': bbx[1] if bbx is not None else None,
            'x_max': bbx[2] if bbx is not None else None,
            'y_max': bbx[3] if bbx is not None else None,
            'real_x': real_position[0] if real_position is not None else None,
            'real_y': real_position[1] if real_position is not None else None,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
    