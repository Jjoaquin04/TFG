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
        
        for ball in ball_list:
            current_ball = [ball['center_x'], ball['center_y']]

            if self.last_ball is None:
                self.last_ball = current_ball
                continue

            if self.last_angle is None:
                Vx = current_ball[0] - self.last_ball[0]
                Vy = current_ball[1] - self.last_ball[1]
                self.last_angle = math.degrees(math.atan2(Vy,Vx))
                
            shot, current_angle = shot_detector.hit_detect(current_ball, self.last_ball,self.last_angle)
            
            if shot:
                nearest_player_id = self._closest_player(current_ball, ball['frame'], player_history)
                
                origen = None
                if nearest_player_id is not None:
                    player_record = player_history.get(nearest_player_id, {}).get(ball['frame'])
                    if player_record:
                        origen = [player_record['real_x'], player_record['real_y']]

                event = Event(
                    impact_frame=ball['frame'],
                    player_id=nearest_player_id,
                    origin_cord=origen
                )
                self.history.append(event)

            self.last_angle = current_angle
            self.last_ball = current_ball

        for i in range(len(self.history) - 1):
            self.history[i].destiny_cord = self.history[i+1].origin_cord

    def _closest_player(self, current_ball, frame_idx, player_history):
        nearest_player = math.inf
        id = None
        for player_id, player_frames in player_history.items():
            record = player_frames.get(frame_idx)
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