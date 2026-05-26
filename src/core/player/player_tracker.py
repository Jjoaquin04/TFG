import cv2
import numpy as np
from core.player.player import Player


class PlayerTracker:
    def __init__(self, homography_matrix):
        self.players: dict[int, Player] = {}
        self.ids_order = []
        self.homography = homography_matrix

    def update(self, ids, boxes, keypoints, frame_idx): # index 0: Top-Left-Player, index 1: Top-Right-Player, index 2: Bottom-Left-Player, index 3: Bottom-Right-Player

        if len(self.ids_order) == 0 and len(boxes) == 4:
            self.add_and_reorder_yoloIds(boxes, ids)
        
        for i in range(len(boxes)):
            player_id = ids[i]
            bbx = boxes[i].tolist()
            kps = keypoints[i].flatten().tolist() if keypoints is not None else []
            
            contact_point = self.get_ground_contact_point(bbx, kps)

            # Apply homography with OpenCV perspectiveTransform
            point_array = np.array([[contact_point]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(point_array, self.homography)
            real_position = (transformed_point[0][0][0], transformed_point[0][0][1])

            if player_id not in self.players:
                self.players[player_id] = Player(id=player_id)
            
            self.players[player_id].update(frame_idx = frame_idx, new_bbx=bbx, new_keypoints=kps, new_real_position=real_position)


    def add_and_reorder_yoloIds(self,boxes, ids):

        for i in range(len(boxes)):
            x_min, _, x_max, y_max = boxes[i]
            (center_x,bottom_y) = (x_min + x_max) / 2 , y_max
            self.ids_order.append({
                'yolo_id': ids[i],
                'center': (center_x,bottom_y)
            })
        
        self.ids_order.sort(key=lambda p: p['center'][1], reverse=True)

        bottom_pair = self.ids_order[:2]
        top_pair = self.ids_order[2:]

        bottom_pair.sort(key=lambda p: p['center'][0])
        top_pair.sort(key=lambda p: p['center'][0])

        self.ids_order = top_pair + bottom_pair

    def get_ground_contact_point(self, bbx, kps):

        """
        Estrategia con fallback:
          1. Promedio de ambos tobillos si ambos son válidos
          2. El tobillo disponible si solo hay uno
          3. Centro inferior del BBX como último recurso
        """
        x_min, y_min, x_max, y_max = bbx
        bbx_bottom_center = ((x_min + x_max) / 2, y_max)

        if not kps or len(kps) < 51:
            return bbx_bottom_center

        # Índices para los tobillos en YOLOv8 pose (15=izquierdo, 16=derecho)
        left_ankle_x, left_ankle_y, left_ankle_conf = kps[45], kps[46], kps[47]
        right_ankle_x, right_ankle_y, right_ankle_conf = kps[48], kps[49], kps[50]
        
        CONF_THRESHOLD = 0.5
        left_valid = left_ankle_conf > CONF_THRESHOLD
        right_valid = right_ankle_conf > CONF_THRESHOLD

        if left_valid and right_valid:
            return ((left_ankle_x + right_ankle_x) / 2, (left_ankle_y + right_ankle_y) / 2)
        elif left_valid:
            return (left_ankle_x, left_ankle_y)
        elif right_valid:
            return (right_ankle_x, right_ankle_y)
        else:
            return bbx_bottom_center

    def get_players_positions(self):
        return [p.current_position() for p in self.players.values()]
    
    def get_players_history(self):
        return [p.history for p in self.players.values()]