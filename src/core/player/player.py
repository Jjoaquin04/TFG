from dataclasses import dataclass
from typing import List

@dataclass
class Player:
    id: int
    bbx: List[float] # Bounding box: [x_min, y_min, x_max, y_max]
    keypoints: List[float] # Lista de keypoints: [x1, y1, x2, y2, ..., xN, yN]
    real_position: List[float] = None 

    @property
    def current_position(self) -> List[float]:
        return self.real_position
    
    def update(self, new_bbx: List[float], new_keypoints: List[float], new_real_position: List[float]):
        self.bbx = new_bbx
        self.keypoints = new_keypoints
        self.real_position = new_real_position
