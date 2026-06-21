import cv2
import numpy as np
import config
from utils import draw_edges_court_connections, draw_comet_tail


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
        self.H_inv = np.linalg.inv(self.mini_homography)
        self.mini_court_points = cv2.perspectiveTransform(config.points_court, self.H_inv)
    
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

        self.start_x = int(frame_width - (self.margin + self.rectangle_width))
        self.start_y = int(self.margin)
        self.end_x = int(frame_width - self.margin)
        self.end_y = int(self.start_y + self.rectangle_height)

    def draw_court(self, frame, player_positions, trajectory_line):
        #1. Dibujar el fondo del minimapa (margen exterior)
        rectangle = frame[self.start_y:self.end_y, self.start_x:self.end_x]
        rectangle_white = np.ones(rectangle.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(rectangle, 0.5, rectangle_white, 0.5, 0) 
        frame[self.start_y:self.end_y, self.start_x:self.end_x] = res

        for position in player_positions:
            #Aplicar homografía inversa para obtener la posición en el mini court
            point_array = np.array([[position]], dtype=np.float32)
            mini_player_position = cv2.perspectiveTransform(point_array, self.H_inv)
        
            mini_player_x = int(np.clip(mini_player_position[0][0][0], self.court_start_x, self.court_end_x))
            mini_player_y = int(np.clip(mini_player_position[0][0][1], self.court_start_y, self.court_end_y))
        
            cv2.circle(frame, (mini_player_x, mini_player_y), radius=5, color=(0, 0, 255), thickness=-1)
                
        if trajectory_line is not None:
            origin_cord, destiny_cord = trajectory_line
            if origin_cord is not None and destiny_cord is not None:
                pt1 = cv2.perspectiveTransform(np.array([[[origin_cord[0], origin_cord[1]]]], dtype=np.float32), self.H_inv)
                pt2 = cv2.perspectiveTransform(np.array([[[destiny_cord[0], destiny_cord[1]]]], dtype=np.float32), self.H_inv)
                
                pt1_x = int(np.clip(pt1[0][0][0], self.court_start_x, self.court_end_x))
                pt1_y = int(np.clip(pt1[0][0][1], self.court_start_y, self.court_end_y))
                pt2_x = int(np.clip(pt2[0][0][0], self.court_start_x, self.court_end_x))
                pt2_y = int(np.clip(pt2[0][0][1], self.court_start_y, self.court_end_y))
                
                frame = draw_comet_tail(frame, (pt1_x, pt1_y), (pt2_x, pt2_y), color=(51, 255, 255), num_points=17)
                
        return draw_edges_court_connections(frame, self.mini_court_points, is_mini_court=True)
