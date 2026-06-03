from core.events import event_tracker
from core import events
import math

class StrokeClassifier:

    def classify_events(self, event_history, players_history, ball_history):
        
        for event in event_history:
            impact_frame = event.get_impact_frame()
            frame_window = event.frames_windows
            player_id = event.get_player_id()
            player_data = players_history.get(player_id)
            if player_data is not None:
                player_info_frame = player_data.get(impact_frame)
                ball_data = ball_history.get(impact_frame)
                mano_pala = self._detect_racket_hand(player_info_frame, ball_data)

                player_keypoints = [
                    player_data.get(frame)['norm_keypoints']
                    for frame in range(frame_window[0], frame_window[1] + 1)
                    if player_data.get(frame) is not None and 'norm_keypoints' in player_data.get(frame)
                ]

    def _detect_racket_hand(self, player_data, ball):
        left_wrist = (player_data['keypoints'][27], player_data['keypoints'][28])
        right_wrist = (player_data['keypoints'][30], player_data['keypoints'][31])
        
        left_distance = math.dist(left_wrist, (ball['center_x'], ball['center_y']))
        right_distance = math.dist(right_wrist, (ball['center_x'], ball['center_y']))

        if left_distance < right_distance:
            return 'left'
        else:
            return 'right'    

    def is_service(self,player_skeleton, racket_hand, event):
        if self.check_line_serve(event.origin_cord) and self.is_cross(event.origin_cord, event.destiny_cord) and self.impact_low_hip(player_skeleton[event.impact_frame], racket_hand) and self.check_soulder_assembly(player_skeleton, event.impact_frame, racket_hand):
            event.type_of_shot = 'service'
            return True
        else:
            return False 
    def check_line_serve(self, event_origin):
        if event_origin is None:
            return False
        if event_origin[0] < 0 and event_origin[1] >= 6.95:
            return True
        elif event_origin[0] > 0 and event_origin[1] >= 6.95:
            return True
        elif event_origin[0] < 0 and event_origin[1] <= -6.95:
            return True
        elif event_origin[0] > 0 and event_origin[1] <= -6.95:
            return True
        else:
            return False

    def is_cross(self, event_origin, event_destiny):
        if event_origin is None or event_destiny is None:
            return False
        if event_origin[0] * event_destiny[0] < 0 and event_origin[1] * event_destiny[1] < 0:
            return True
        else:
            return False
    
    def impact_low_hip(self, norm_keypoints, racket_hand):
        wrist_idx = 9 if racket_hand == 'left' else 10
        return norm_keypoints[wrist_idx][1] > -15.0
    
    def check_soulder_assembly(self, norm_keypoints_window, impact_frame, racket_hand):
        wrist_idx = 9 if racket_hand == 'left' else 10
        shoulder_idx =  5 if racket_hand == 'left' else 6

        for frame_idx in range(len(norm_keypoints_window)):
            keypoints_wrist = norm_keypoints_window[frame_idx][wrist_idx][1]
            keypoints_shoulder = norm_keypoints_window[frame_idx][shoulder_idx][1]

            if keypoints_wrist <= keypoints_shoulder:
                return True
        return False
            
            