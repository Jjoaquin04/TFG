import math
def detect_racket_hand(player_data, ball_center):
    
        left_wrist = (player_data['keypoints'][9][0], player_data['keypoints'][9][1])
        right_wrist = (player_data['keypoints'][10][0], player_data['keypoints'][10][1])
        
        left_distance = math.dist(left_wrist, (ball_center[0], ball_center[1]))
        right_distance = math.dist(right_wrist, (ball_center[0], ball_center[1]))
        
        if left_distance < right_distance:
            return 'left', left_distance
        else:
            return 'right', right_distance