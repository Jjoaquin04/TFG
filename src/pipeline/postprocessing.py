import pandas as pd
import json
import os
import config
from core import EventTracker, StrokeClassifier

def postprocessing(raw_json_path: str):

    raw_path = raw_json_path
    file_name = os.path.basename(raw_path)
    interp_path = os.path.join(config.INTERP_JSON_PATH, "interpolated_json", file_name)
    
    print(f"Loading raw detections from {raw_path}...")
    with open(raw_path, 'r') as f:
        data = json.load(f)
        
    ball_history = data.get('ball', [])
    players_history = data.get('players', {})
    
    if not ball_history:
        print("No ball data found.")
        return None
    
    interpolated_ball = interpolate_ball(ball_history)
    players_history = calculate_players_centers(players_history)
    
    event_tracker = EventTracker()
    event_tracker.track(interpolated_ball, players_history)
    # stroke_classifier = StrokeClassifier()
    # events = stroke_classifier.classify_events(event_tracker.get_history(), players_history, interpolated_ball)
    events = event_tracker.get_history()
    
    events_dict = {}
    if events:
        for e in events:
            frame_key = str(e.impact_frame)
            if frame_key not in events_dict:
                events_dict[frame_key] = []
            # Serialize dataclass to dict, drop complex properties
            event_data = {
                'impact_frame': e.impact_frame,
                'player_id': e.player_id,
                'type_of_shot': getattr(e, 'type_of_shot', None),
                'trajectory': getattr(e, 'trajectory', None),
                'origin_cord': e.origin_cord,
                'destiny_cord': e.destiny_cord
            }
            events_dict[frame_key].append(event_data)
            
    # Reasignar para que el json resultante tenga los centers de la bola y jugadores
    data['ball'] = interpolated_ball
    data['players'] = players_history
    data['events'] = events_dict
    os.makedirs(os.path.dirname(interp_path), exist_ok=True)
    with open(interp_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Interpolation finished! Saved cleaned data to {interp_path}")
    return interp_path

def interpolate_ball(ball_df):
    df_ball = pd.DataFrame(ball_df).set_index('frame')
    df_ball_interp = df_ball.interpolate(method='linear', limit_direction='both', limit=3)
    
    df_ball_interp = calculate_centers(df_ball_interp) 
    df_ball_interp = df_ball_interp.where(pd.notna(df_ball_interp), None)
    return df_ball_interp.reset_index().to_dict(orient='records')

def calculate_players_centers(dict_players):
    for _, player_data in dict_players.items():
        for record in player_data.values():
            record['center_x'] = (record['x_min'] + record['x_max']) / 2
            record['center_y'] = (record['y_min'] + record['y_max']) / 2
    return dict_players
        
def calculate_centers(dataframe):
    dataframe['center_x'] = (dataframe['x_min'] + dataframe['x_max']) / 2
    dataframe['center_y'] = (dataframe['y_min'] + dataframe['y_max']) / 2
    return dataframe
