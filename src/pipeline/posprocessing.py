import pandas as pd
import json
import os
import config
from core import EventTracker

def posprocessing(raw_json_path: str):

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
    
    data['ball'] = interpolate_ball(ball_history)
    
    event_tracker = EventTracker()
    event_tracker.track(data['ball'], players_history)

    os.makedirs(os.path.dirname(interp_path), exist_ok=True)
    with open(interp_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Interpolation finished! Saved cleaned data to {interp_path}")
    return interp_path

def interpolate_ball(ball_df):
    df_ball = pd.DataFrame(ball_df).set_index('frame')
    df_ball_interp = df_ball.interpolate(method='linear', limit_direction='both', limit=5)
    
    # Restaura los floats en un formato seguro para JSON de nuevo a lista de diccionarios
    return df_ball_interp.reset_index().to_dict(orient='records')



