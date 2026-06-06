import pandas as pd
import json
import numpy as np
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
    
    players_history = filter_players(players_history)
    ball_history = filter_ball_outliers(ball_history)
    
    #interpolated_ball = interpolate_ball(ball_history)
    players_history = calculate_players_centers(players_history)
    
    """
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
    """
    # Reasignar para que el json resultante tenga los centers de la bola y jugadores
    data['ball'] = ball_history
    data['players'] = players_history
    #data['events'] = events_dict
    os.makedirs(os.path.dirname(interp_path), exist_ok=True)
    with open(interp_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Interpolation finished! Saved cleaned data to {interp_path}")
    return interp_path

def interpolate_ball(ball_df):
    df_copy = ball_df.copy()
    nan_size = group_nan(df_copy)

    df_ball_interp = ball_df.interpolate(method='linear', limit_direction='both')
    is_big = nan_size > 8

    df_ball_interp.loc[is_big, ['x_min', 'y_min', 'x_max', 'y_max','center_x', 'center_y']] = None
    return df_ball_interp

    #return df_ball_interp.reset_index().to_dict(orient='records')

def calculate_players_centers(dict_players):
    for _, player_data in dict_players.items():
        for record in player_data.values():
            record['center_x'] = (record['x_min'] + record['x_max']) / 2
            record['center_y'] = (record['y_min'] + record['y_max']) / 2
    return dict_players
        
def calculate_centers(data_ball):
    df_ball = pd.DataFrame(data_ball).set_index('frame')
    df_ball['center_x'] = (df_ball['x_min'] + df_ball['x_max']) / 2
    df_ball['center_y'] = (df_ball['y_min'] + df_ball['y_max']) / 2
    return df_ball

def filter_players(players_history):
    player_lengths = {p_id: len(frames) for p_id, frames in players_history.items()}
    top_4_ids = sorted(player_lengths, key=player_lengths.get, reverse=True)[:4]

    filtered_players = {p_id: players_history[p_id] for p_id in top_4_ids}
    return filtered_players

def group_nan(ball_frame: pd.DataFrame):
    nan_groups= ball_frame['x_min'].notna().cumsum()
    solo_nans = nan_groups[ball_frame['x_min'].isna()]
    len_consecutive_nan = solo_nans.groupby(solo_nans).size()
    
    len_map = nan_groups.map(len_consecutive_nan).fillna(0)
    return len_map

def filter_ball_outliers(ball_history, window_size=7, threshold_dist=50):
    
    df_ball = calculate_centers(ball_history)
    df_copy = df_ball.copy()

    median_x = df_copy['center_x'].rolling(window=window_size, center=True, min_periods=1).median()
    median_y = df_copy['center_y'].rolling(window=window_size, center=True, min_periods=1).median()

    deviation = np.sqrt(((median_x - df_copy['center_x'])**2) + ((median_y - df_copy['center_y'])**2))
    atipic = deviation > threshold_dist
    df_copy.loc[atipic, ['x_min', 'y_min', 'x_max', 'y_max','center_x', 'center_y']] = None

    df_copy = interpolate_ball(df_copy)

    return df_copy