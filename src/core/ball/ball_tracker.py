import cv2
import numpy as np

from core.ball.ball import Ball


class BallTracker:

    def __init__(self, homography_matrix):
        self.ball = Ball()
        self.homography = homography_matrix

    def update(self, ball_bbx):
        (center_x, center_y) = (ball_bbx[0] + ball_bbx[2]) / 2 , (ball_bbx[1] + ball_bbx[3]) / 2
        transformed_point = cv2.perspectiveTransform(np.array([[[center_x,center_y]]], dtype=np.float32), self.homography)
        real_position = (transformed_point[0][0][0], transformed_point[0][0][1])
        self.ball.update(ball_bbx,real_position)
    
    def get_ball_position(self):
        return self.ball.get_real_position()