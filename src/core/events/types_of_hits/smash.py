import math
from utils.event.event_utils import check_soulder_assembly, impact_high

def is_smash(player_keypoints_window, impact_keypoints, racket_hand, event, next_event, ball_history):
        cond_armado = check_soulder_assembly(player_keypoints_window, racket_hand)
        cond_impacto_alto = impact_high(impact_keypoints, racket_hand)
        
        score = (cond_armado + cond_impacto_alto) / 2.0
        
        # Exclusión mutua: un remate va hacia abajo (Y en pantalla aumenta).
        # Si la pelota sube claramente (Y disminuye) en los primeros frames, penalizamos fuertemente.
        is_going_up = False
        if next_event is not None:
            y_start = None
            min_y_after = math.inf
            frames_to_check = min(3, next_event.impact_frame - event.impact_frame)
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
        wrist_y = None
        shoulder_y = None
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

        print(f"      [is_smash] armado_hombro={cond_armado}, impacto_alto={cond_impacto_alto}, is_going_up={is_going_up} | wrist_y={wrist_y}, shoulder_y={shoulder_y}")
        print(f"    Event {event.impact_frame}, [SCORE smash] {score, tie_breaker}\n")
        return [score, 'smash', tie_breaker]