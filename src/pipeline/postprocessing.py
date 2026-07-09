import pandas as pd
import json
import numpy as np
import os
import config
from core import EventTracker, StrokeClassifier
from utils import remove_static_players, filter_ball_outliers, interpolate_ball, remove_false_detections
from utils import get_ground_contact_point, normalice_keypoints, apply_homography, calculate_players_centers, remove_static_players, reorder_yolo_ids


def postprocessing(raw_json_path: str):

    raw_path = raw_json_path
    file_name = os.path.basename(raw_path)
    interp_path = os.path.join(config.INTERP_JSON_PATH, "interpolated_json", file_name)
    
    print(f"Loading raw detections from {raw_path}...")
    with open(raw_path, 'r') as f:
        data = json.load(f)
        
    if not data.get('ball'):
        print("No ball data found.")
        return None
        
    # 1. Fase de Limpieza
    data = _clean_data(data)
    
    # 2. Fase de Extracción de Datos
    data = _extract_features(data)
    
    # 3. Clasificación de Eventos
    interpolated_ball_list = data['ball']
    players_history = data['players']
    
    
    interpolated_ball_dict = {b['frame']: b for b in interpolated_ball_list}
    event_tracker = EventTracker()
    event_tracker.track(interpolated_ball_dict, players_history)
    
    stroke_classifier = StrokeClassifier()
    events = stroke_classifier.classify_events(event_tracker.get_history(), players_history, interpolated_ball_dict)
    events = event_tracker.get_history()
    
    # Asignar la mano de la pala dominante al diccionario general de cada jugador
    players_hands = stroke_classifier.get_players_racket_hands()
    for pid, hand in players_hands.items():
        str_pid = str(pid)
        if str_pid in players_history:
            players_history[str_pid]['racket_hand'] = hand
            
    events_dict = {}
    if events:
        for e in events:
            frame_key = str(e.impact_frame)
            if frame_key not in events_dict:
                events_dict[frame_key] = []
            event_data = {
                'impact_frame': e.impact_frame,
                'player_id': e.player_id,
                'type_of_shot': getattr(e, 'type_of_shot', None),
                'trajectory': getattr(e, 'trajectory', None),
                'origin_cord': e.origin_cord,
                'destiny_cord': e.destiny_cord
            }
            events_dict[frame_key].append(event_data)
            
    data['events'] = events_dict
    os.makedirs(os.path.dirname(interp_path), exist_ok=True)
    with open(interp_path, 'w') as f:
        json.dump(data, f, indent=2,cls= config.NumpyEncoder)
        
    print(f"Interpolation finished! Saved cleaned data to {interp_path}")
    return interp_path
    

def _clean_data(data):
    ball_history = data.get('ball', [])
    
    if ball_history:
        ball_history = filter_ball_outliers(ball_history)
        data['ball'] = interpolate_ball(ball_history)
        
    return data

def _extract_features(data):
    players_history = data.get('players', {})
    
    if not players_history:
        return data
        
    players_history = calculate_players_centers(players_history)

    # Extraer la matriz de homografía (el usuario la guarda en la posición 1 de court_information)
    court_info = data.get('court', [])
    if len(court_info) > 1:
        H = np.array(court_info[1], dtype=np.float32)
    else:
        H = None
        
    #Diccionario -> DataFrame(Solo contiene informacion de la data del jugador)
    records = []
    for pid, frames in players_history.items():
        for f_id, f_data in frames.items():
            records.append(f_data)
            
    players_df = pd.DataFrame(records)
    
    #Funciones de extraccion
    players_df = remove_static_players(players_df, 300.0)
    players_df = remove_false_detections(players_df, 5)
    players_df = get_ground_contact_point(players_df)
    players_df = normalice_keypoints(players_df)
    players_df = reorder_yolo_ids(players_df)
    
    
    if H is not None:
        players_df = apply_homography(players_df, H, 'contact_x', 'contact_y')
        
    if 'ball' in data and data['ball'] and H is not None:
        ball_df = pd.DataFrame(data['ball'])
        ball_df = apply_homography(ball_df, H, 'center_x', 'center_y')
        data['ball'] = ball_df.to_dict(orient='records')
        
    #Volver a meter en un dict
    players_df = players_df.replace({np.nan: None})
    players_history = {}
    for player_id, p_df in players_df.groupby('player_id'):
        str_pid = str(player_id)
        record = p_df.to_dict(orient='records')
        players_history[str_pid] = {str(int(row['frame'])): row for row in record}
                
    data['players'] = players_history

    primer_frame = players_df['frame'].min()

    jugadores_primer_frame = players_df[players_df['frame'] == primer_frame]
    return data