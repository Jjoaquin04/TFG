from dataclasses import dataclass
from typing import List

@dataclass
class Player:
    id: int
    bbx: List[float] # Bounding box: [x_min, y_min, x_max, y_max]
    keypoints: List[float] # Lista de keypoints: [x1, y1, x2, y2, ..., xN, yN]

    @property
    def current_position(self) -> List[float,float]:
        # Devuelve el centro del bounding box como posición actual del jugador
        x_min, _, x_max, y_max = self.bbx
        center_x = (x_min + x_max) / 2
        bottom_y = y_max
        return [center_x, bottom_y]
    
    def update(self, new_bbx: List[float], new_keypoints: List[float]):
        self.bbx = new_bbx
        self.keypoints = new_keypoints
    
