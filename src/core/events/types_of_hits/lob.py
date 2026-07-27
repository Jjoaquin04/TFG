import math
from utils.event.event_utils import impact_high

def is_lob(event, next_event, ball_history, impact_keypoints, racket_hand):
        if next_event is None:
            print(f"      [is_lob] No hay next_event. Score=0.0")
            print(f"    Event {event.impact_frame}, [SCORE Lob] (0.0, 0.0)\n")
            return [0.0, 'lob', 0.0]
            
        frames_flight = next_event.impact_frame - event.impact_frame
        
        # Buscamos la Y mínima de la pelota durante el vuelo (punto más alto en la imagen)
        min_y = math.inf
        for f in range(event.impact_frame, next_event.impact_frame):
            b = ball_history.get(str(f)) or ball_history.get(f)
            if b and b.get('center_y') is not None and not math.isnan(b.get('center_y')):
                if b['center_y'] < min_y:
                    min_y = b['center_y']
                    
        # Buscar alturas de inicio y fin de vuelo para comprobar si hace parábola
        start_y = None
        for f in range(event.impact_frame, min(next_event.impact_frame, event.impact_frame + 5)):
            b = ball_history.get(str(f)) or ball_history.get(f)
            if b and b.get('center_y') is not None and not math.isnan(b.get('center_y')):
                start_y = b['center_y']
                break
                
        end_y = None
        for f in range(next_event.impact_frame, max(event.impact_frame, next_event.impact_frame - 5) - 1, -1):
            b = ball_history.get(str(f)) or ball_history.get(f)
            if b and b.get('center_y') is not None and not math.isnan(b.get('center_y')):
                end_y = b['center_y']
                break
                
        is_parabola = False
        if start_y is not None and end_y is not None:
            highest_player_y = min(start_y, end_y)
            # Exigimos que la pelota suba al menos 60 píxeles por encima del punto más alto (menor Y) de los jugadores
            if min_y < (highest_player_y - 60):
                is_parabola = True
        else:
            # Fallback si no tenemos coordenadas
            if min_y < 250:
                is_parabola = True
                
        cond1 = frames_flight > 30
        cond2 = (frames_flight > 20 and is_parabola)
        
        score = 0.0
        if cond1 and cond2:
            score = 1.0
        elif cond1 or cond2:
            score = 0.5
            
        # Si no describe una parábola real (solo es un golpe lento), lo penalizamos fuertemente
        if not is_parabola:
            score -= 0.5
            
        # Exclusión mutua: un globo no se golpea con la técnica de remate (muñeca alta)
        is_high_impact = False
        if impact_keypoints is not None and racket_hand is not None:
            is_high_impact = impact_high(impact_keypoints, racket_hand)
            
        if is_high_impact:
            score *= 0.25 # Penalización severa
            
        # Tie breaker: combinar tiempo de vuelo alto y min_y bajo
        tie_breaker = 0.0
        if min_y != math.inf:
            tie_breaker = frames_flight + max(0, 350.0 - min_y)
            
        print(f"      [is_lob] frames_flight={frames_flight}, min_y={min_y:.2f} | cond1(>30)={cond1}, cond2(>25 & y<250)={cond2} | is_high_impact={is_high_impact}")
        print(f"    Event {event.impact_frame}, [SCORE Lob] {score, tie_breaker}\n")
        return [score, 'lob', tie_breaker]