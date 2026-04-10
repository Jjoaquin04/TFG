import cv2
import numpy as np


import config
from utils.image_video.ImageVideoHandler import draw_edges_court_connections


class MiniCourt():

    def __init__(self, frame_width):

        self.rectangle_width = 250
        self.rectangle_height = 450
        self.margin = 50
        self.padding = 30
    
        self.set_background_court(frame_width)
        self.set_mini_court()
        self.mini_homography = self._build_mini_homography()
        
        # Invertimos la homografía para mapear de coordanadas reales a coordenadas en pixeles del mini court
        H_inv = np.linalg.inv(self.mini_homography)
        self.mini_court_points = cv2.perspectiveTransform(config.points_court, H_inv)
    
    def set_mini_court(self):
        
        self.court_start_x = self.start_x + self.padding
        self.court_start_y = self.start_y + self.padding
        self.court_end_x = self.end_x - self.padding
        self.court_end_y = self.end_y - self.padding

    def _build_mini_homography(self):

        src_points = np.array([
            [self.court_start_x, self.court_start_y], # Esquina Superior Izquierda
            [self.court_end_x, self.court_start_y], # Esquina Superior Derecha
            [self.court_start_x, self.court_end_y], # Esquina Inferior Izquierda
            [self.court_end_x, self.court_end_y] # Esquina Inferior Derecha
        ], dtype=np.float32)


        dst_points = np.array([
            [config.points_court[7][0][0], config.points_court[7][0][1]], # Esquina Superior Izquierda
            [config.points_court[8][0][0], config.points_court[8][0][1]], # Esquina Superior Derecha
            [config.points_court[9][0][0], config.points_court[9][0][1]], # Esquina Inferior Izquierda
            [config.points_court[10][0][0], config.points_court[10][0][1]]  # Esquina Inferior Derecha
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        return H
    

    def set_background_court(self, frame_width):

        self.start_x = frame_width - (self.margin + self.rectangle_width)
        self.start_y = self.margin
        self.end_x = frame_width - self.margin
        self.end_y = self.start_y + self.rectangle_height

    def draw_court(self, frame):
        # 1. Dibujar el fondo del minimapa (margen exterior)
        rectangle = frame[self.start_y:self.end_y, self.start_x:self.end_x]
        rectangle_white = np.ones(rectangle.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(rectangle, 0.5, rectangle_white, 0.5, 0) 
        frame[self.start_y:self.end_y, self.start_x:self.end_x] = res
             
        return draw_edges_court_connections(frame, self.mini_court_points, is_mini_court=True)
