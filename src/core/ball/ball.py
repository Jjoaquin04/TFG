from dataclasses import dataclass, field
from typing import List
import pandas as pd

@dataclass
class Ball:

    history: List[dict] = field(default_factory=list)
    bbx: List[float] = None 

    def update(self,frame_idx: int,  bbx: List[float]):
        self.bbx = bbx
        self.history.append({
            'frame': frame_idx,
            'x_min': bbx[0] if bbx is not None else None,
            'y_min': bbx[1] if bbx is not None else None,
            'x_max': bbx[2] if bbx is not None else None,
            'y_max': bbx[3] if bbx is not None else None,
        })
    