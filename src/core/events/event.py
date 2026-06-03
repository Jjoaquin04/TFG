from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    impact_frame: int = None
    player_id: int = None
    type_of_shot: str = None
    trajectory: str = None
    origin_cord: List[float] = None
    destiny_cord: List[float] = None

    RANGE_FRAMES = 10
    
    @property
    def frames_windows(self):
        initial_frame = max(0,self.impact_frame - self.RANGE_FRAMES)
        last_frame = self.impact_frame + self.RANGE_FRAMES

        return [initial_frame, last_frame]

    def get_player_id(self):
        return self.player_id

    def get_impact_frame(self):
        return self.impact_frame