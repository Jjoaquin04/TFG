import math
from utils.event.event_utils import impact_high

def is_drive_or_volley(event, next_event, ball_history, impact_keypoints, racket_hand):
        score = 0.65 # Base score robusto para ser el golpe por defecto si no es nada extremo
        min_y = math.inf
        
        # Penalizamos fuertemente si sube mucho (exclusión con globo)
        if next_event is not None:
            frames_to_check = min(30, next_event.impact_frame - event.impact_frame)
            for f in range(event.impact_frame, event.impact_frame + frames_to_check):
                b = ball_history.get(str(f)) or ball_history.get(f)
                if b and b.get('center_y') is not None and not math.isnan(b.get('center_y')):
                    if b['center_y'] < min_y:
                        min_y = b['center_y']
                        
        if min_y < 250:
            score *= 0.5 # Le quitamos puntos porque se parece a un globo
            
        # Penalizamos si se armó muy por encima del hombro (exclusión con smash)
        is_high_impact = False
        if impact_keypoints is not None and racket_hand is not None:
            is_high_impact = impact_high(impact_keypoints, racket_hand)
            
        if is_high_impact:
            score *= 0.25
            
        # Diferenciamos Drive vs Volley según la posición en la pista y altura de impacto
        is_volley = False
        
        # Obtenemos la altura de la muñeca respecto a la cadera si es posible
        wrist_high = False
        if impact_keypoints is not None and racket_hand is not None:
            wrist_idx = 9 if racket_hand == 'left' else 10
            hip_idx = 23 if racket_hand == 'left' else 24 # MediaPipe hips are 23, 24
            
            # Como fallback, si no tenemos las caderas de mediapipe, usamos un umbral estático
            # Pero en normalized keypoints, los valores son relativos al centro.
            if len(impact_keypoints) > hip_idx:
                wrist_y = impact_keypoints[wrist_idx][1]
                hip_y = impact_keypoints[hip_idx][1]
                if wrist_y != 0.0 and hip_y != 0.0:
                    # Y negativo es hacia arriba. Si la muñeca está por encima de la cadera (menor valor)
                    if wrist_y < hip_y - 0.08:
                        wrist_high = True

        if event.origin_cord is not None:
            abs_y = abs(event.origin_cord[1])
            # Si está en la zona clara de volea, le damos prioridad sobre el remate
            if abs_y < 4.5:
                if is_high_impact:
                    score = 0.85 # Prioridad sobre remate débil, pero cede ante uno claro
                else:
                    score = 1.5 # Volea clara
                is_volley = True
            # Volea a media pista (zona de saque)
            elif abs_y < 7.0 and (wrist_high or is_high_impact):
                is_volley = True
                
        hit_type = 'volley' if is_volley else 'drive'
        
        # Tie breaker: Preferimos el evento con menor distancia a la bola
        tie_breaker = 1.0 / (1.0 + event.score) if event.score is not None else 0.0

        return [score, hit_type, tie_breaker]