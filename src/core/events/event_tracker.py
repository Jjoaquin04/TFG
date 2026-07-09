from ast import Dict
import math
import core.events.direction_change_detector as direction_detector
from .event import Event
from utils.event.event_utils import calculate_min_hand_distance
from typing import List

class EventTracker:

    def __init__(self):
        self.history: List[Event] = []
        self.last_event_frame = -999

    def track(self, ball_dict, player_history):
        # --- CLuster NMS ---
        candidate_cluster = []
        CLUSTER_MAX_GAP = 7  # Frames máximos sin meter nada en el cluster

        last_ball = None
        last_valid_frame = None
        last_angle = None

        for ball in ball_dict.values():
            current_ball = [ball['center_x'], ball['center_y']]
            print(f"frame: {ball['frame']} -> {current_ball[0], current_ball[1]}\n")
            if candidate_cluster and (ball['frame'] - candidate_cluster[-1]['frame'] > CLUSTER_MAX_GAP):
                if last_valid_frame and (last_valid_frame - candidate_cluster[-1]['frame'] <= 5):
                    print(f"  -> [NMS] Cluster previo DESCARTADO por solaparse con inicio de oclusión (en gap check).")
                    candidate_cluster.clear()
                else:
                    self._process_cluster(candidate_cluster)

            if math.isnan(current_ball[0]) or math.isnan(current_ball[1]):
                continue
                
            if last_ball is None:
                last_ball = current_ball 
                last_valid_frame = ball['frame']
                continue

            none_count = ball['frame'] - last_valid_frame
            #print(f"{none_count}\n")
            
            if none_count > 1:
                print(f"[Frame {ball['frame']}] Oclusión finalizada. Duración (none_count): {none_count} frames. (Y pre-oclusión: {last_ball[1]:.1f})")
                pre_player, pre_dist = self._closest_player(last_ball, last_valid_frame, player_history)
                if none_count > 25 or last_ball[1] < 100:
                    print(f"  -> DESCARTADO por GLOBO/FUERA (Duración > 30 o Y_pre < 100)")
                    Vx = current_ball[0] - last_ball[0]
                    Vy = current_ball[1] - last_ball[1]
                    last_angle = math.degrees(math.atan2(Vy, Vx))
                    last_ball = current_ball
                    last_valid_frame = ball['frame']
                    continue
                else:
                    if candidate_cluster and (last_valid_frame - candidate_cluster[-1]['frame'] <= 5):
                        print(f"  -> [NMS] Cluster previo DESCARTADO por solaparse con inicio de oclusión.")
                        candidate_cluster.clear()
                    else:
                        self._process_cluster(candidate_cluster)
                        
                    possible_event, new_angle = direction_detector.detect_direction_change(current_ball, last_ball, last_angle)
                    last_angle = new_angle
                    if possible_event:
                        print(f"  -> CAMBIO ÁNGULO DETECTADO tras oclusión -> {new_angle}")
                        closest_player, nearest_distance = self._closest_player(current_ball, ball['frame'], player_history)
                        
                        #Comparar si la distancia pre oclusion es menor o igual a la de despues de la oclusion
                        if pre_dist <= nearest_distance:
                            best_player = pre_player #El jugador es el de antes de la oclusion
                            best_dist = pre_dist
                            print(f"  -> Se asigna a jugador PRE-OCLUSIÓN {best_player} (Dist: {best_dist:.1f} px)")
                        else:
                            best_player = closest_player #El jugador es el de después de la oclusion
                            best_dist = nearest_distance
                            print(f"  -> Se asigna a jugador POST-OCLUSIÓN {best_player} (Dist: {best_dist:.1f} px)")

                        if best_dist < 300.0:
                            impact_frame_estimated = last_valid_frame + (none_count // 2) #Frame de impacto estimado la mitad de la oclusion
                            
                            # Obtener coordenadas de origen desde el jugador
                            player_record = player_history.get(best_player, {}).get(str(impact_frame_estimated))
                            origin_cord = None
                            if player_record and player_record.get('real_x') is not None and not math.isnan(player_record.get('real_x')):
                                origin_cord = [player_record['real_x'], player_record['real_y']]

                            if impact_frame_estimated - self.last_event_frame >= 15:
                                event = Event(
                                    impact_frame=impact_frame_estimated,
                                    player_id=best_player,
                                    score=best_dist, 
                                    origin_cord = origin_cord
                                )
                                print(f"  -> [!] EVENTO OCLUSIÓN AÑADIDO (Frame estimado: {impact_frame_estimated})")
                                self.history.append(event)
                                self.last_event_frame = impact_frame_estimated
                            else:
                                print(f"  -> IGNORADO: Cooldown activo en oclusión.")
                        else:
                            print(f"  -> DESCARTADO: Ningún jugador estaba suficientemente cerca en los extremos de la oclusión.")
            else:
                #Flujo normal, posible golpeo sin oclusion
                possible_event, new_angle = direction_detector.detect_direction_change(current_ball, last_ball, last_angle)
                last_angle = new_angle
                if possible_event: 
                    print(f"[Frame {ball['frame']}] CAMBIO ÁNGULO DETECTADO en flujo normal: {new_angle}")
                    closest_player, nearest_distance = self._closest_player(current_ball, ball['frame'], player_history)
                    print(f"  -> Distancia a jugador {closest_player}: {nearest_distance:.1f} px (Umbral Normal: 100.0)")
                    if nearest_distance < 100.0:
                        origin_cord = None
                        player_record = player_history.get(closest_player, {}).get(str(ball['frame']))
                        if player_record and player_record.get('real_x') is not None and not math.isnan(player_record.get('real_x')):
                            origin_cord = [player_record['real_x'], player_record['real_y']]
                            
                        candidate_cluster.append({
                            'frame': ball['frame'],
                            'distance': nearest_distance,
                            'player_id': closest_player, 
                            'origin_cord': origin_cord
                        })
                        print(f"  -> Añadido a candidato NMS temporal (Dist: {nearest_distance:.1f}).")
                    else:
                        print(f"  -> DESCARTADO: Jugador muy lejos (posible bote o pared).")
                        
            last_ball = current_ball
            last_valid_frame = ball['frame']
            
        #Ejecutar el cluster si existe cuando acabamos
        self._process_cluster(candidate_cluster)

        # --- ASIGNAR DESTINY CORD ---
        for i in range(len(self.history)):
            current_event = self.history[i]
            if i < len(self.history) - 1:
                next_event = self.history[i+1]
                current_event.destiny_cord = next_event.origin_cord
            else:
                # Para el último evento, buscamos la última posición válida de la pelota
                last_ball = None
                for ball in reversed(list(ball_dict.values())):
                    if ball.get('real_x') is not None and not math.isnan(ball.get('real_x')):
                        last_ball = ball
                        break
                if last_ball:
                    current_event.destiny_cord = [last_ball['real_x'], last_ball['real_y']]

    def _process_cluster(self, candidate_cluster):
        if not candidate_cluster:
            return
            
        #Penalizacion con 1.2px por frame con respecto al primero del cluster para evitar que si tras el impacto
        #la pelota sigue cerca del jugador pueda tomarse como el momento del golpeo
        start_frame = candidate_cluster[0]['frame']
        for c in candidate_cluster:
            frames_late = c['frame'] - start_frame
            c['score'] = c['distance'] + (frames_late * 1.2)  
            
        #Frame con menor distancia (score)
        best = min(candidate_cluster, key=lambda x: x['score'])

        #Comprobacion de que por angulo y posicion de los jugadores en golpeos cruzados un jugador no puede hacer un evento si justo el anterior es su compañero
        if self.history:
            last_event = self.history[-1]
            last_hitter = int(float(last_event.player_id))
            current_hitter = int(float(best['player_id']))
            frames_since_last = best['frame'] - last_event.impact_frame
            
            # Comparamos si son del mismo equipo (Nuestros roles son 0 y 1 para arriba, 2 y 3 para abajo)
            is_partner = (last_hitter in [0, 1] and current_hitter in [0, 1]) or \
                         (last_hitter in [2, 3] and current_hitter in [2, 3])
            
            #Si es el compañero o él mismo, y ha pasado "poco tiempo"
            if is_partner and frames_since_last < 40:
                last_score = getattr(last_event, 'score', 999.0)
                
                if current_hitter == last_hitter:
                    # Mismo jugador
                    if frames_since_last < 15:
                        # Demasiado pronto: es el efecto "perspective drag" o jitter visual del golpe anterior.
                        print(f"  -> [NMS] DESCARTADO MISMO JUGADOR: frames={frames_since_last} < 15. Efecto de perspectiva 2D o jitter.")
                    else:
                        # Caso típico de Bote -> Golpe de pared.
                        # Siempre sobreescribimos asumiendo que el último evento es el golpe real, porque tras un golpe válido la pelota se aleja del jugador.
                        print(f"  -> [NMS] OVERWRITE MISMO JUGADOR: Actualizamos golpe de {current_hitter} (Score {last_score:.1f} -> {best['score']:.1f}).")
                        last_event.impact_frame = best['frame']
                        last_event.score = best['score']
                        self.last_event_frame = best['frame']
                else:
                    #Compañero, caso típico de tiro cruzado.
                    print(f"  -> [NMS] DESCARTADO: Jugador {current_hitter} es compañero de {last_hitter} en tiro cruzado.")
                        
                candidate_cluster.clear()
                return
                
        event = Event(
            impact_frame=best['frame'],
            player_id=best['player_id'],
            origin_cord=best.get('origin_cord'),
            score=best['score']
        )
        print(f"  -> [NMS] EVENTO AÑADIDO (Frame: {best['frame']}, Dist Real: {best['distance']:.1f} px, Score: {best['score']:.1f}) elegido de un cluster de {len(candidate_cluster)} frames.")
        self.history.append(event)
        self.last_event_frame = best['frame']
        candidate_cluster.clear()

    def _closest_player(self, current_ball, frame_idx, player_history):
        nearest_distance = math.inf
        id = None
        for player_id, player_frames in player_history.items():
            record = player_frames.get(str(frame_idx))
            if not record:
                continue
                
            point_ball = [current_ball[0], current_ball[1]]
            distance = calculate_min_hand_distance(record, point_ball)
            if distance < nearest_distance:
                id = player_id
                nearest_distance = distance

        return id, nearest_distance

    def get_history(self):
        return self.history