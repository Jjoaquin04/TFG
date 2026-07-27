import math
from utils.event.event_utils import is_cross, impact_low_hip

def is_service(player_keypoints_window, impact_keypoints, racket_hand, event, prev_event_raw, players_history, impact_frame, frames_cortes, is_first_event):
        c1 = _check_line_serve(event.origin_cord)
        c2 = is_cross(event.origin_cord, event.destiny_cord, event)
        c3 = impact_low_hip(impact_keypoints, racket_hand)
        c4 = _check_players_position_serve(event, players_history, impact_frame)
        condition_list = [c1, c2, c3, c4]
        
        sum_cond = sum(condition_list)
        score = sum_cond / len(condition_list)

        is_after_cut = False
        if is_first_event:
            is_after_cut = True
        else:
            for cut_frame in frames_cortes:
                print(f"{impact_frame - cut_frame}")
                if 0 <= (impact_frame - cut_frame) <= 45:
                    is_after_cut = True
                    break
                    
        # Tratamiento de dobles saques tras corte o errores:
        es_mismo_jugador = True
        is_second_serve = False
        if prev_event_raw is not None and prev_event_raw.type_of_shot == 'service':
            if prev_event_raw.player_id != event.player_id:
                es_mismo_jugador = False
            else:
                is_second_serve = True
                
        if not es_mismo_jugador:
            # Si el evento anterior fue un saque y ahora golpea otro jugador, es IMPOSIBLE que esto sea un saque.
            return [0.0, 'service', 0.0]
            
        # Bonificación si es después de un corte (o primer evento del partido)
        if is_after_cut:
            score += 1.0
        # Penalización si no es después de un corte Y no es un segundo saque (mismo jugador tras saque)
        elif not is_second_serve:
            score -= 1.0
            
        # Tie breaker: Distancia ideal desde la línea de saque (Y=6.95)
        tie_breaker = 0.0
        if event.origin_cord is not None:
            dist_to_line = abs(abs(event.origin_cord[1]) - 6.95)
            tie_breaker = 1.0 / (1.0 + dist_to_line)

        print(f"      [is_service] Conditions: detras_linea={c1}, cruzado={c2}, impacto_bajo={c3}, pos_equipo={c4} | after_cut={is_after_cut} | mismo_jug={es_mismo_jugador}")
        print(f"    Event {event.impact_frame}, [SCORE Servicio] {score, tie_breaker}\n")
        return [score, 'service', tie_breaker]

def _check_players_position_serve(event, players_history, impact_frame):
        try:
            server_id = int(float(event.player_id))
        except (ValueError, TypeError):
            return False

        if server_id in [0, 1]:
            partner_id = 1 if server_id == 0 else 0
            opponents_ids = [2, 3]
        elif server_id in [2, 3]:
            partner_id = 3 if server_id == 2 else 2
            opponents_ids = [0, 1]
        else:
            return False
            
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

def _check_line_serve(event_origin):
        if event_origin is None:
            return False
        # Para el debug imprimimos las coords
        print(f"[StrokeClassifier] check_line_serve coords: X={event_origin[0]:.2f}, Y={event_origin[1]:.2f}")
        # La línea de saque está en Y = 6.95 y Y = -6.95. Dejamos 5.5 como margen generoso hacia delante.
        return abs(event_origin[1]) >= 5.5