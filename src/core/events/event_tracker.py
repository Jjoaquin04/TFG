import math
import core.events.shot_detector as shot_detector
from event import Event
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
                nearest_player_id = self._closest_player(current_ball, player_history)
                event = Event(
                    impact_frame=ball['frame'],
                    player_id=nearest_player_id,
                )
                
                self.history.append(event)

            self.last_angle = current_angle
            self.last_ball = current_ball

    def _closest_player(current_ball, player_history):
        nearest_player = math.inf
        id = None
        for player in player_history.values():
            point_ball = (current_ball[0], current_ball[1])
            point_player = (player['center_x'], player['center_y'])
            distance = abs((math.dist(point_ball, point_player)))
            if distance < nearest_player:
                id = player['player_id']
                nearest_player = distance

        return id

