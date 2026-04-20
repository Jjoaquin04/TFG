from core.ball.ball import Ball


class BallTracker:

    def __init__(self, homography_matrix):
        self.ball = Ball()
        self.homography_matrix = homography_matrix

    def update(self, ball_bbx):
        self.ball.update_position(ball_bbx)