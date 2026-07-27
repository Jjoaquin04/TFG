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
            
        # Diferenciamos Drive vs Volley según la posición en la pista
        is_volley = False
        if event.origin_cord is not None:
            # Si el jugador está en la zona de volea (más cerca de la red, ej: Y entre -4 y 4)
            if abs(event.origin_cord[1]) < 4.5:
                is_volley = True
                
        hit_type = 'volley' if is_volley else 'drive'
        
        # Tie breaker: Preferimos el evento con menor distancia a la bola
        tie_breaker = 1.0 / (1.0 + event.score) if event.score is not None else 0.0

        print(f"      [is_{hit_type}] min_y={min_y:.2f}, is_high_impact={is_high_impact} | origin_y={(event.origin_cord[1] if event.origin_cord else 0.0):.2f}")
        print(f"    Event {event.impact_frame}, [SCORE {hit_type.capitalize()}] {score, tie_breaker}\n")
        return [score, hit_type, tie_breaker]