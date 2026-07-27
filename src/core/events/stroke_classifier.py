from .event_tracker import EventTracker
from .event import Event
from .types_of_hits.service import is_service
from .types_of_hits.smash import is_smash
from .types_of_hits.lob import is_lob
from .types_of_hits.volley_drive import is_drive_or_volley
import math
import numpy as np

class StrokeClassifier:

    def __init__(self):
        self.racket_player: dict[int, (int,int)] = {}

    def classify_events(self, events_history, players_history, ball_history, frames_cortes=[]):
        for i, event in enumerate(events_history):
            impact_frame = event.get_impact_frame()
            frame_window = event.frames_windows
            player_id = event.get_player_id()
            player_data = players_history.get(player_id)
            prev_event = events_history[i-1] if i > 0 else None
            next_event = events_history[i+1] if i + 1 < len(events_history) else None
            
            #Validar que prev_event y next_event pertenecen al mismo punto (no hay corte entre medias)
            if prev_event is not None:
                for cut in frames_cortes:
                    if prev_event.impact_frame < cut <= impact_frame:
                        prev_event = None
                        break
            if next_event is not None:
                for cut in frames_cortes:
                    if impact_frame < cut <= next_event.impact_frame:
                        next_event = None
                        break
            
            if player_data is not None:
                player_info_frame = player_data.get(str(impact_frame))
                
                # Recopilar la ventana de keypoints
                player_keypoints_window = [
                    player_data.get(str(frame))['norm_keypoints']
                    for frame in range(frame_window[0], frame_window[1] + 1)
                    if player_data.get(str(frame)) is not None and 'norm_keypoints' in player_data.get(str(frame))
                ]
                
                impact_keypoints = player_info_frame['norm_keypoints'] if player_info_frame and 'norm_keypoints' in player_info_frame else None
                if impact_keypoints is not None and len(player_keypoints_window) > 0:
                    racket_hand = self.detect_racket_hand_by_distance(frame_window, player_data, ball_history, player_id)
                    
                    scores = []
                    
                    #Le pasamos el evento anterior para comprobar si es un doble saque
                    prev_event_raw = events_history[i-1] if i > 0 else None
                    scores.append(is_service(player_keypoints_window, impact_keypoints, racket_hand, event, prev_event_raw, players_history, impact_frame, frames_cortes, i == 0))
                    scores.append(is_smash(player_keypoints_window, impact_keypoints, racket_hand, event, next_event, ball_history))
                    scores.append(is_lob(event, next_event, ball_history, impact_keypoints, racket_hand))
                    scores.append(is_drive_or_volley(event, next_event, ball_history, impact_keypoints, racket_hand))
                    
                    # score_info is [score, event_type, tie_breaker_score]
                    # Ordenar por score descendente y en caso de empate por tie_breaker_score descendente
                    scores.sort(key=lambda x: (x[0], x[2]), reverse=True)
                    
                    best_match = scores[0]
                    if best_match[0] > 0.0:
                        event.type_of_shot = best_match[1]
                        print(f"[StrokeClassifier] EVENTO {event.impact_frame} CLASIFICADO COMO {best_match[1].upper()} (Score: {best_match[0]:.2f}, TieBreaker: {best_match[2]:.2f}).")
                    else:
                        event.type_of_shot = 'unknown'

        return events_history    

    def detect_racket_hand_by_distance(self, frame_window, player_data, ball_history, player_id):
        min_left_dist = math.inf
        min_right_dist = math.inf
        
        for frame in range(frame_window[0], frame_window[1] + 1):
            p_frame = player_data.get(str(frame))
            b_frame = ball_history.get(frame) or ball_history.get(str(frame))
            
            if p_frame and b_frame and 'keypoints' in p_frame:
                kp = p_frame['keypoints']
                bx, by = b_frame.get('center_x'), b_frame.get('center_y')
                
                if bx is None or math.isnan(bx) or len(kp) <= 10:
                    continue
                
                left_wrist = kp[9]
                right_wrist = kp[10]
                
                if left_wrist[0] != 0.0 or left_wrist[1] != 0.0:
                    dist_l = math.dist((left_wrist[0], left_wrist[1]), (bx, by))
                    if dist_l < min_left_dist:
                        min_left_dist = dist_l
                        
                if right_wrist[0] != 0.0 or right_wrist[1] != 0.0:
                    dist_r = math.dist((right_wrist[0], right_wrist[1]), (bx, by))
                    if dist_r < min_right_dist:
                        min_right_dist = dist_r

        print(f"[StrokeClassifier] Distancia min a la bola -> Izq: {min_left_dist:.1f}, Der: {min_right_dist:.1f}")
        hand = 'left' if min_left_dist < min_right_dist else 'right'
        if player_id in self.racket_player:
            (left_count, right_count) = self.racket_player.get(player_id)
            if hand == 'left':
                left_count = left_count + 1
            else:
                right_count = right_count + 1

            self.racket_player[player_id] = (left_count, right_count)    
        else:
            self.racket_player[player_id] = (1, 0) if hand == 'left' else (0, 1)

        return hand

    def get_players_racket_hands(self):
        result = {}
        for pid, counts in self.racket_player.items():
            left_count, right_count = counts
            result[pid] = 'left' if left_count > right_count else 'right'
        return result