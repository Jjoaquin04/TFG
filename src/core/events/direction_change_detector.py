import math

def detect_direction_change(current_ball, last_ball, last_angle):
    Vx = current_ball[0] - last_ball[0]
    Vy = current_ball[1] - last_ball[1]
    
    angle = math.degrees(math.atan2(Vy,Vx))
    
    if last_angle is None:
        return False, angle
        
    angle_change = abs(angle - last_angle)
    if angle_change > 180: 
        angle_change = 360 - angle_change
    if angle_change > 60:
        return True, angle
    
    return False, angle