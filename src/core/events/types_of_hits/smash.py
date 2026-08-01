import math
from utils.event.event_utils import check_soulder_assembly, impact_high

def is_smash(player_keypoints_window, impact_keypoints, racket_hand, event, next_event, ball_history):
        cond_armado = check_soulder_assembly(player_keypoints_window, racket_hand)
        is_high_impact_cond = impact_high(impact_keypoints, racket_hand)
        
        score = 0.0
        if cond_armado and is_high_impact_cond:
            score = 1.0
        elif cond_armado:
            score = 0.8  # Bandeja (armado claro pero impacto medio)
        elif is_high_impact_cond:
            score = 0.7  # Asegura ganar al drive incluso si falla is_going_down
        
        is_going_down = False
        if next_event is not None:
            y_start = None
            min_y_after = math.inf
            max_y_after = -math.inf
            frames_to_check = min(3, next_event.impact_frame - event.impact_frame)
            for f in range(event.impact_frame, event.impact_frame + frames_to_check):
                b = ball_history.get(str(f)) or ball_history.get(f)
                if b and b.get('center_y') is not None and not math.isnan(b.get('center_y')):
                    if y_start is None:
                        y_start = b['center_y']
                    if b['center_y'] < min_y_after:
                        min_y_after = b['center_y']
                    if b['center_y'] > max_y_after:
                        max_y_after = b['center_y']
                        
            if y_start is not None:
                y_end = None
                b_end = ball_history.get(str(next_event.impact_frame)) or ball_history.get(next_event.impact_frame)
                if b_end and b_end.get('center_y') is not None and not math.isnan(b_end.get('center_y')):
                    y_end = b_end['center_y']

                is_top_court = True
                if y_end is not None:
                    is_top_court = (y_end > y_start)

                if is_top_court:
                    if max_y_after > y_start + 5: # Comprobar trayectoria descendente
                        is_going_down = True
                else:
                    # Jugador en la pista inferior: la pelota debe subir en la pantalla (la Y disminuye)
                    if min_y_after < y_start - 5:
                        is_going_down = True
                
        if is_going_down:
            score += 0.3  # Bonificación por trayectoria descendente
            
        # Tie breaker: Distancia ideal de altura respecto al hombro.
        tie_breaker = 0.0
        wrist_y = None
        shoulder_y = None
        wrist_idx = 9 if racket_hand == 'left' else 10
        shoulder_idx = 5 if racket_hand == 'left' else 6
        if impact_keypoints and len(impact_keypoints) > max(wrist_idx, shoulder_idx):
            if impact_keypoints[wrist_idx][0] != 0.0 and impact_keypoints[wrist_idx][1] != 0.0:
                wrist_y = impact_keypoints[wrist_idx][1]
                shoulder_y = impact_keypoints[shoulder_idx][1]
                
                # Ajuste: el impacto alto acepta wrist_y <= shoulder_y + 15.0
                high_threshold = shoulder_y + 15.0
                diff = high_threshold - wrist_y # Cuanto menor es wrist_y, mayor es diff (mejor)
                
                if diff > 0:
                    tie_breaker = diff

        
        return [score, 'smash', tie_breaker]