import math

def calculate_min_hand_distance(player_data, ball_center):
    left_wrist = (player_data['keypoints'][9][0], player_data['keypoints'][9][1])
    right_wrist = (player_data['keypoints'][10][0], player_data['keypoints'][10][1])
    
    left_distance = math.dist(left_wrist, (ball_center[0], ball_center[1]))
    right_distance = math.dist(right_wrist, (ball_center[0], ball_center[1]))
    
    return min(left_distance, right_distance)

def is_cross(event_origin, event_destiny, event):
        if event_origin is None or event_destiny is None:
            return False
        if event_origin[0] * event_destiny[0] < 0 and event_origin[1] * event_destiny[1] < 0:
            event.trajectory = 'cross'
            return True
        else:
            event.trajectory = 'parallel'
            return False

def impact_high(norm_keypoints, racket_hand):
        wrist_idx = 9 if racket_hand == 'left' else 10
        shoulder_idx = 5 if racket_hand == 'left' else 6
        
        if norm_keypoints[wrist_idx][0] == 0.0 and norm_keypoints[wrist_idx][1] == 0.0:
            return False 
            
        wrist_y = norm_keypoints[wrist_idx][1]
        shoulder_y = norm_keypoints[shoulder_idx][1]
        
        # El impacto es alto si la muñeca está por encima del hombro o muy cerca
        # Recordar que Y negativo es hacia arriba. Así que wrist_y <= shoulder_y + 15.0
        return wrist_y <= shoulder_y + 15.0

def impact_low_hip(norm_keypoints, racket_hand):
        wrist_idx = 9 if racket_hand == 'left' else 10
        if norm_keypoints[wrist_idx][0] == 0.0 and norm_keypoints[wrist_idx][1] == 0.0:
            return True # Si está ocluido, asumimos que sí (o no lo penalizamos)
        
        wrist_y = norm_keypoints[wrist_idx][1]
        return wrist_y > -30.0

def check_soulder_assembly(norm_keypoints_window, racket_hand):
        wrist_idx = 9 if racket_hand == 'left' else 10
        shoulder_idx =  5 if racket_hand == 'left' else 6
        
        min_wrist_y = math.inf
        ref_y_at_min = 0

        for frame_idx in range(len(norm_keypoints_window)):
            keypoints_wrist_x = norm_keypoints_window[frame_idx][wrist_idx][0]
            keypoints_wrist_y = norm_keypoints_window[frame_idx][wrist_idx][1]
            keypoints_ref_y = norm_keypoints_window[frame_idx][shoulder_idx][1] # Comparamos con el hombro

            if keypoints_wrist_x != 0.0 or keypoints_wrist_y != 0.0:
                if keypoints_wrist_y < min_wrist_y:
                    min_wrist_y = keypoints_wrist_y
                    ref_y_at_min = keypoints_ref_y
                    
                # Si la muñeca supera al hombro (armado claro para remate/bandeja)
                if keypoints_wrist_y <= keypoints_ref_y:
                    return True
                    
        return False