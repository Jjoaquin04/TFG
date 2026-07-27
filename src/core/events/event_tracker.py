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

    def track(self, ball_dict, player_history, frames_cortes=[]):
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
                if last_valid_frame and (last_valid_frame - candidate_cluster[-1]['frame'] <= 2):
                    print(f"  -> [NMS] Cluster previo DESCARTADO por solaparse con inicio de oclusión (en gap check).")
                    candidate_cluster.clear()
                else:
                    self._process_cluster(candidate_cluster, frames_cortes)

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
                    if candidate_cluster and (last_valid_frame - candidate_cluster[-1]['frame'] <= 2):
                        print(f"  -> [NMS] Cluster previo DESCARTADO por solaparse con inicio de oclusión.")
                        candidate_cluster.clear()
                    else:
                        self._process_cluster(candidate_cluster, frames_cortes)
                        
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
                    print(f"  -> Distancia a jugador {closest_player}: {nearest_distance:.1f} px (Umbral Normal: 130.0)")
                    if nearest_distance < 130.0:
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
        self._process_cluster(candidate_cluster, frames_cortes)

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

    def _process_cluster(self, candidate_cluster, frames_cortes=[]):
        if not candidate_cluster:
            return
            
        #Penalizacion con 1.2px por frame con respecto al primero del cluster
        start_frame = candidate_cluster[0]['frame']
        for c in candidate_cluster:
            frames_late = c['frame'] - start_frame
            c['score'] = c['distance'] + (frames_late * 1.2)  
            
        # Ordenamos los candidatos de menor a mayor score
        candidate_cluster.sort(key=lambda x: x['score'])
        
        best = None
        for candidate in candidate_cluster:
            current_hitter = int(float(candidate['player_id']))
            is_valid = True
            
            if self.history:
                last_event = self.history[-1]
                last_hitter = int(float(last_event.player_id))
                
                # Check for cuts between last event and this candidate
                has_cut = False
                for cut in frames_cortes:
                    if last_event.impact_frame < cut <= candidate['frame']:
                        has_cut = True
                        break
                        
                if not has_cut:
                    # Comparamos si son del mismo equipo
                    is_partner = (last_hitter in [0, 1] and current_hitter in [0, 1]) or \
                                 (last_hitter in [2, 3] and current_hitter in [2, 3])
                    
                    if is_partner:
                        frames_since_last = candidate['frame'] - last_event.impact_frame
                        if current_hitter == last_hitter:
                            if frames_since_last < 15:
                                print(f"  -> [NMS] DESCARTADO MISMO JUGADOR: frames={frames_since_last} < 15. Efecto perspectiva.")
                                is_valid = False
                            elif frames_since_last < 40:
                                # Caso típico de Bote -> Golpe de pared.
                                last_score = getattr(last_event, 'score', 999.0)
                                print(f"  -> [NMS] OVERWRITE MISMO JUGADOR: Actualizamos golpe de {current_hitter} (Score {last_score:.1f} -> {candidate['score']:.1f}).")
                                last_event.impact_frame = candidate['frame']
                                last_event.score = candidate['score']
                                self.last_event_frame = candidate['frame']
                                candidate_cluster.clear()
                                return
                            else:
                                # No hay corte, y es el mismo jugador, pero pasaron mas de 40 frames
                                print(f"  -> [NMS] DESCARTADO JUGADOR {current_hitter}: Es de su equipo y no ha habido corte (Ilegal en padel).")
                                is_valid = False
                        else:
                            # Compañero
                            print(f"  -> [NMS] DESCARTADO COMPAÑERO {current_hitter}: No ha habido corte (Ilegal en padel).")
                            is_valid = False

            if is_valid:
                best = candidate
                break

        if not best:
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