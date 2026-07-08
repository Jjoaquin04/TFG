import cv2
import numpy as np
from .player import Player


class PlayerTracker:
    def __init__(self):
        self.players: dict[int, Player] = {}

    def update(self, frame, ids, boxes, keypoints, frame_idx): 
        
        for i in range(len(boxes)):
            player_id = int(ids[i])
            bbx = boxes[i].tolist()
            kps = keypoints[i].tolist() if keypoints is not None else []
            
            if player_id not in self.players:
                self.players[player_id] = Player(id=player_id)
            
            self.players[player_id].update(frame=frame, frame_idx=frame_idx, new_bbx=bbx, new_keypoints=kps)
    
    def get_players_history(self):
        history = {}
        for player_id, p in self.players.items():
            history[player_id] = p.history
         
        return history