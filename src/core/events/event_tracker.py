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

    def track(self, ball_dict, player_history, cut_frames=[]):
        # Agrupación de detecciones cercanas (Cluster NMS)
        candidate_cluster = []
        CLUSTER_MAX_GAP = 7  # Máximo de frames sin añadir elementos al cluster

        last_ball = None
        last_valid_frame = None
        last_angle = None

        for ball in ball_dict.values():
            current_ball = [ball['center_x'], ball['center_y']]
            if candidate_cluster and (ball['frame'] - candidate_cluster[-1]['frame'] > CLUSTER_MAX_GAP):
                if last_valid_frame and (last_valid_frame - candidate_cluster[-1]['frame'] <= 2):
                    candidate_cluster.clear()
                else:
                    self._process_cluster(candidate_cluster, cut_frames)

            if math.isnan(current_ball[0]) or math.isnan(current_ball[1]):
                continue
                
            if last_ball is None:
                last_ball = current_ball 
                last_valid_frame = ball['frame']
                continue

            none_count = ball['frame'] - last_valid_frame
            
            if none_count > 1:
                pre_player, pre_dist = self._closest_player(last_ball, last_valid_frame, player_history)
                if none_count > 25 or last_ball[1] < 100:
                    Vx = current_ball[0] - last_ball[0]
                    Vy = current_ball[1] - last_ball[1]
                    last_angle = math.degrees(math.atan2(Vy, Vx))
                    last_ball = current_ball
                    last_valid_frame = ball['frame']
                    continue
                else:
                    if candidate_cluster and (last_valid_frame - candidate_cluster[-1]['frame'] <= 2):
                        candidate_cluster.clear()
                    else:
                        self._process_cluster(candidate_cluster, cut_frames)

                    possible_event, new_angle = direction_detector.detect_direction_change(current_ball, last_ball, last_angle)
                    last_angle = new_angle
                    if possible_event:
                        closest_player, nearest_distance = self._closest_player(current_ball, ball['frame'], player_history)
                        
                        # Comparamos si la distancia antes de la oclusión es menor o igual a la de después
                        if pre_dist <= nearest_distance:
                            best_player = pre_player # El jugador es el de antes de la oclusión
                            best_dist = pre_dist
                        else:
                            best_player = closest_player # El jugador es el de después de la oclusión
                            best_dist = nearest_distance

                        if best_dist < 300.0:
                            impact_frame_estimated = last_valid_frame + (none_count // 2) # Frame de impacto estimado (mitad de la oclusión)
                            
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
                                self.history.append(event)
                                self.last_event_frame = impact_frame_estimated
            else:
                # Flujo normal, posible golpeo sin oclusión
                possible_event, new_angle = direction_detector.detect_direction_change(current_ball, last_ball, last_angle)
                last_angle = new_angle
                if possible_event: 
                    closest_player, nearest_distance = self._closest_player(current_ball, ball['frame'], player_history)
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
                        
                        
            last_ball = current_ball
            last_valid_frame = ball['frame']
            
        # Ejecutar el cluster remanente si existe al terminar
        self._process_cluster(candidate_cluster, cut_frames)

        # Asignar coordenadas de destino
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

    def _process_cluster(self, candidate_cluster, cut_frames=[]):
        if not candidate_cluster:
            return
            
        # Penalización temporal: 1.2px por frame de retraso respecto al primero del cluster
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
                
                # Revisar si hay un corte entre el ultimo evento y este candidato
                has_cut = False
                for cut in cut_frames:
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
                                is_valid = False
                            elif frames_since_last < 40:
                                # Caso típico de Bote -> Golpe de pared.
                                last_score = getattr(last_event, 'score', 999.0)
                                last_event.impact_frame = candidate['frame']
                                last_event.score = candidate['score']
                                self.last_event_frame = candidate['frame']
                                candidate_cluster.clear()
                                return
                            else:
                                # No hay corte, y es el mismo jugador, pero pasaron más de 40 frames
                                is_valid = False
                        else:
                            # Compañero
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