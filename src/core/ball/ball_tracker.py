import cv2
import numpy as np
from .ball import Ball


class BallTracker:

    def __init__(self):
        self.ball = Ball()

    def update(self, ball_detections, frame_idx):
        if len(ball_detections) == 0:
            self.ball.update(frame_idx, [])
        else:
            self.ball.update(frame_idx, ball_detections)
    
    def get_ball_history(self):
        return self.ball.history