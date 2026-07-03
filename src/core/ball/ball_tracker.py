import cv2
import numpy as np
from .ball import Ball


class BallTracker:

    def __init__(self):
        self.ball = Ball()

    def update(self, ball_bbx, frame_idx):
        if ball_bbx is None:
            self.ball.update(frame_idx,None)
        else:
            self.ball.update(frame_idx, ball_bbx)
    
    def get_ball_history(self):
        return self.ball.history