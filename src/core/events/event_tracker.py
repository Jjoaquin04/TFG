from ast import Dict
import math
import core.events.shot_detector as shot_detector
from .event import Event
from typing import List

class EventTracker:

    def __init__(self):
        self.last_ball = None
        self.last_angle = None
        self.history: List[Event] = []

    def track(self, ball_list, player_history):
        last_position_ball = None
        last_valid_vector_x = None
        last_valid_vector_y = None
        last_frame = None # Ultimo frame donde perdimos la bola 
        for ball in ball_list:

            if ball.get('center_x') is None or ball.get('center_y') is None:
                if last_position_ball is not None and last_frame is None:
                    last_frame = ball['frame']
            else:
                current_ball = [ball['center_x'], ball['center_y']]
                if last_position_ball is None:
                    last_position_ball = current_ball
                    self.last_ball = current_ball
                    continue

                Vx = current_ball[0] - last_position_ball[0]
                Vy = current_ball[1] - last_position_ball[1]
                
                came_from_gap = False
                # --- PASO 3: LÓGICA DE OCLUSIÓN ACTIVA ---
                if last_frame is not None:
                    came_from_gap = True
                    if last_valid_vector_x is not None:
                        # Si se invierte la X, asumimos impacto durante la oclusión
                        if (Vx * last_valid_vector_x) < 0:
                            print(f"¡Oclusión Activa! Golpe detectado entre {last_frame} y {ball['frame']}")
                            impact_frame = last_frame + (ball['frame'] - last_frame) // 2
                            nearest_player_id = self._closest_player(current_ball, impact_frame, player_history)
                            
                            origen = None
                            if nearest_player_id is not None:
                                player_record = player_history.get(nearest_player_id, {}).get(str(impact_frame))
                                if player_record:
                                    origen = [player_record.get('real_x'), player_record.get('real_y')]

                                event = Event(
                                    impact_frame=impact_frame,
                                    player_id=nearest_player_id,
                                    origin_cord=origen
                                )
                                self.history.append(event)
                    
                    last_frame = None

                # --- LÓGICA NORMAL ---
                # Si venimos de un hueco (globo o tapada) reseteamos la referencia del ángulo para no confundir a la lógica normal por culpa de la gravedad
                if self.last_angle is None or came_from_gap:
                    self.last_angle = math.degrees(math.atan2(Vy,Vx))
                    shot = False
                    current_angle = self.last_angle
                else:
                    shot, current_angle = shot_detector.shot_detect(current_ball, self.last_ball, self.last_angle)
                
                if shot:
                    nearest_player_id = self._closest_player(current_ball, ball['frame'], player_history)
                    
                    origen = None
                    if nearest_player_id is not None:
                        player_record = player_history.get(nearest_player_id, {}).get(str(ball['frame']))
                        if player_record:
                            # Filtramos paredes: distancia máxima de 500 px
                            dist = math.dist((current_ball[0], current_ball[1]), (player_record['center_x'], player_record['center_y']))
                            if dist < 500:
                                origen = [player_record.get('real_x'), player_record.get('real_y')]
                                event = Event(
                                    impact_frame=ball['frame'],
                                    player_id=nearest_player_id,
                                    origin_cord=origen
                                )
                                self.history.append(event)

                last_position_ball = current_ball
                last_valid_vector_x = Vx
                last_valid_vector_y = Vy
                self.last_angle = current_angle
                self.last_ball = current_ball

        for i in range(len(self.history) - 1):
            self.history[i].destiny_cord = self.history[i+1].origin_cord

    def _closest_player(self, current_ball, frame_idx, player_history):
        nearest_player = math.inf
        id = None
        for player_id, player_frames in player_history.items():
            record = player_frames.get(str(frame_idx))
            if not record:
                continue
                
            point_ball = (current_ball[0], current_ball[1])
            point_player = (record['center_x'], record['center_y'])
            distance = abs((math.dist(point_ball, point_player)))
            if distance < nearest_player:
                id = player_id
                nearest_player = distance

        return id

    def get_history(self):
        return self.history