from core.events import event_tracker
from core import events
import math
import numpy as np

class StrokeClassifier:

    def __init__(self):
        self.racket_player: dict[int, (int,int)] = {}

    def classify_events(self, events_history, players_history, ball_history):

        for i, event in enumerate(events_history):
            impact_frame = event.get_impact_frame()
            frame_window = event.frames_windows
            player_id = event.get_player_id()
            player_data = players_history.get(player_id)
            next_event = events_history[i+1] if i + 1 < len(events_history) else None
            
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
                    
                    scores.append(self.is_service(player_keypoints_window, impact_keypoints, racket_hand, event, players_history, impact_frame))
                    scores.append(self.is_smash(player_keypoints_window, impact_keypoints, racket_hand, event, next_event, ball_history))
                    scores.append(self.is_lob(event, next_event, ball_history, impact_keypoints, racket_hand))
                    
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

    def is_service(self, player_keypoints_window, impact_keypoints, racket_hand, event, players_history, impact_frame):
        condition_list = [ 
            self.check_line_serve(event.origin_cord), 
            self.is_cross(event.origin_cord, event.destiny_cord, event),
            self.impact_low_hip(impact_keypoints, racket_hand),
            self.check_players_position_serve(event, players_history, impact_frame)
        ]
        
        sum_cond = sum(condition_list)
        score = sum_cond / len(condition_list)

        # Tie breaker: Distancia ideal desde la línea de saque (Y=6.95)
        tie_breaker = 0.0
        if event.origin_cord is not None:
            dist_to_line = abs(abs(event.origin_cord[1]) - 6.95)
            tie_breaker = 1.0 / (1.0 + dist_to_line)

        print(f"    Event {event.impact_frame}, [SCORE Servicio] {score, tie_breaker}\n")
        return [score, 'service', tie_breaker]

    def check_players_position_serve(self, event, players_history, impact_frame):
        # En padel profesional los jugadores son 1-2 (equipo 1) y 3-4 (equipo 2)
        try:
            server_id = int(event.player_id)
        except (ValueError, TypeError):
            return False

        if server_id in [1, 2]:
            partner_id = 2 if server_id == 1 else 1
            opponents_ids = [3, 4]
        else:
            partner_id = 4 if server_id == 3 else 3
            opponents_ids = [1, 2]
            
        partner_data = players_history.get(partner_id) or players_history.get(str(partner_id))
        opp1_data = players_history.get(opponents_ids[0]) or players_history.get(str(opponents_ids[0]))
        opp2_data = players_history.get(opponents_ids[1]) or players_history.get(str(opponents_ids[1]))
        
        conditions_met = 0
        
        if partner_data:
            frame_data = partner_data.get(str(impact_frame)) or partner_data.get(impact_frame)
            if frame_data and frame_data.get('real_y') is not None and not math.isnan(frame_data.get('real_y')):
                if abs(frame_data['real_y']) < 4.0: # Compañero cerca de la red
                    conditions_met += 1
                    
        for opp in [opp1_data, opp2_data]:
            if opp:
                frame_data = opp.get(str(impact_frame)) or opp.get(impact_frame)
                if frame_data and frame_data.get('real_y') is not None and not math.isnan(frame_data.get('real_y')):
                    if abs(frame_data['real_y']) > 5.0: # Oponentes cerca del fondo
                        conditions_met += 1
                        
        # Si cumple al menos 2 condiciones (compañero en la red + 1 oponente al fondo, o 2 oponentes al fondo)
        return conditions_met >= 2

    def is_smash(self, player_keypoints_window, impact_keypoints, racket_hand, event, next_event, ball_history):
        cond_armado = self.check_soulder_assembly(player_keypoints_window, racket_hand)
        cond_impacto_alto = self.impact_high(impact_keypoints, racket_hand)
        
        score = (cond_armado + cond_impacto_alto) / 2.0
        
        # Exclusión mutua: un remate va hacia abajo (Y en pantalla aumenta).
        # Si la pelota sube claramente (Y disminuye) en los primeros frames, penalizamos fuertemente.
        is_going_up = False
        if next_event is not None:
            y_start = None
            min_y_after = math.inf
            frames_to_check = min(10, next_event.impact_frame - event.impact_frame)
            for f in range(event.impact_frame, event.impact_frame + frames_to_check):
                b = ball_history.get(str(f)) or ball_history.get(f)
                if b and b.get('center_y') is not None and not math.isnan(b.get('center_y')):
                    if y_start is None:
                        y_start = b['center_y']
                    if b['center_y'] < min_y_after:
                        min_y_after = b['center_y']
                        
            if y_start is not None and min_y_after < y_start - 15: # Subió al menos 15 px
                is_going_up = True
                
        if is_going_up:
            score *= 0.25 # Penalización severa
            
        # Tie breaker: Distancia ideal de altura respecto al hombro.
        tie_breaker = 0.0
        wrist_idx = 9 if racket_hand == 'left' else 10
        shoulder_idx = 5 if racket_hand == 'left' else 6
        if impact_keypoints and len(impact_keypoints) > max(wrist_idx, shoulder_idx):
            if impact_keypoints[wrist_idx][0] != 0.0 and impact_keypoints[wrist_idx][1] != 0.0:
                wrist_y = impact_keypoints[wrist_idx][1]
                shoulder_y = impact_keypoints[shoulder_idx][1]
                
                # Ajuste: el impacto alto acepta wrist_y <= shoulder_y + 15.0
                umbral_alto = shoulder_y + 15.0
                diff = umbral_alto - wrist_y # Cuanto menor es wrist_y, mayor es diff (mejor)
                
                if diff > 0:
                    tie_breaker = diff

        print(f"    Event {event.impact_frame}, [SCORE smash] {score, tie_breaker}\n")
        return [score, 'smash', tie_breaker]

    def is_lob(self, event, next_event, ball_history, impact_keypoints, racket_hand):
        if next_event is None:
            return [0.0, 'lob', 0.0]
            
        frames_flight = next_event.impact_frame - event.impact_frame
        
        # Buscamos la Y mínima de la pelota durante el vuelo (punto más alto en la imagen)
        min_y = math.inf
        for f in range(event.impact_frame, next_event.impact_frame):
            b = ball_history.get(str(f)) or ball_history.get(f)
            if b and b.get('center_y') is not None and not math.isnan(b.get('center_y')):
                if b['center_y'] < min_y:
                    min_y = b['center_y']
                    
        cond1 = frames_flight > 30
        cond2 = (frames_flight > 25 and min_y < 250)
        
        score = 0.0
        if cond1 and cond2:
            score = 1.0
        elif cond1 or cond2:
            score = 0.5
            
        # Exclusión mutua: un globo no se golpea con la técnica de remate (muñeca alta)
        is_high_impact = False
        if impact_keypoints is not None and racket_hand is not None:
            is_high_impact = self.impact_high(impact_keypoints, racket_hand)
            
        if is_high_impact:
            score *= 0.25 # Penalización severa
            
        # Tie breaker: combinar tiempo de vuelo alto y min_y bajo
        tie_breaker = 0.0
        if min_y != math.inf:
            tie_breaker = frames_flight + max(0, 350.0 - min_y)
            
        print(f"    Event {event.impact_frame}, [SCORE Lob] {score, tie_breaker}\n")
        return [score, 'lob', tie_breaker]

    def check_line_serve(self, event_origin):
        if event_origin is None:
            return False
        # Para el debug imprimimos las coords
        print(f"[StrokeClassifier] check_line_serve coords: X={event_origin[0]:.2f}, Y={event_origin[1]:.2f}")
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

    def is_cross(self, event_origin, event_destiny, event):
        if event_origin is None or event_destiny is None:
            return False
        if event_origin[0] * event_destiny[0] < 0 and event_origin[1] * event_destiny[1] < 0:
            event.trajectory = 'cross'
            return True
        else:
            event.trajectory = 'parallel'
            return False
    
    def impact_high(self, norm_keypoints, racket_hand):
        wrist_idx = 9 if racket_hand == 'left' else 10
        shoulder_idx = 5 if racket_hand == 'left' else 6
        
        if norm_keypoints[wrist_idx][0] == 0.0 and norm_keypoints[wrist_idx][1] == 0.0:
            return False 
            
        wrist_y = norm_keypoints[wrist_idx][1]
        shoulder_y = norm_keypoints[shoulder_idx][1]
        
        # El impacto es alto si la muñeca está por encima del hombro o muy cerca
        # Recordar que Y negativo es hacia arriba. Así que wrist_y <= shoulder_y + 15.0
        return wrist_y <= shoulder_y + 15.0

    def impact_low_hip(self, norm_keypoints, racket_hand):
        wrist_idx = 9 if racket_hand == 'left' else 10
        if norm_keypoints[wrist_idx][0] == 0.0 and norm_keypoints[wrist_idx][1] == 0.0:
            return True # Si está ocluido, asumimos que sí (o no lo penalizamos)
        
        wrist_y = norm_keypoints[wrist_idx][1]
        return wrist_y > -30.0
    
    def check_soulder_assembly(self, norm_keypoints_window, racket_hand):
        wrist_idx = 9 if racket_hand == 'left' else 10
        shoulder_idx =  5 if racket_hand == 'left' else 6
        
        min_wrist_y = math.inf
        ref_y_at_min = 0

        for frame_idx in range(len(norm_keypoints_window)):
            keypoints_wrist_x = norm_keypoints_window[frame_idx][wrist_idx][0]
            keypoints_wrist_y = norm_keypoints_window[frame_idx][wrist_idx][1]
            keypoints_ref_y = norm_keypoints_window[frame_idx][shoulder_idx][1] # Comparamos con el hombro

            if keypoints_wrist_x != 0.0 or keypoints_wrist_y != 0.0:
                if keypoints_wrist_y < min_wrist_y:
                    min_wrist_y = keypoints_wrist_y
                    ref_y_at_min = keypoints_ref_y
                    
                # Si la muñeca supera al hombro (armado claro para remate/bandeja)
                if keypoints_wrist_y <= keypoints_ref_y:
                    return True
                    
        return False
            
            