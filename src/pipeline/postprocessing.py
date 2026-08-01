import pandas as pd
import json
import numpy as np
import os
import cv2
import config
from tqdm import tqdm
from core import EventTracker, StrokeClassifier, KeypointsCourt
from utils import remove_static_players, remove_false_detections_players, remove_false_detections_ball, filter_ball_outliers, get_ground_contact_point, normalice_keypoints, apply_homography, calculate_players_centers, reorder_yolo_ids

def postprocessing(raw_json_path: str, video_path: str):

    raw_path = raw_json_path
    file_name = os.path.basename(raw_path)
    interp_path = os.path.join(config.INTERP_JSON_PATH, "interpolated_json", file_name)
    
    print(f"Loading raw detections from {raw_path}...")
    with open(raw_path, 'r') as f:
        data = json.load(f)
        
    if not data.get('ball'):
        print("No ball data found.")
        return None
        
    court_info = data.get('court', [])
    if len(court_info) > 1:
        H = np.array(court_info[1], dtype=np.float32)
        court_corners = np.array(court_info[0], dtype=np.float32)
        
        # 0. Fase de ajuste interactivo de la pista
        print("\nAbriendo ventana de ajuste interactivo de la pista...")
        kc = KeypointsCourt()
        kc.keypoints = court_corners
        kc.extract_rest_of_kpoints()
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            full_keypoints, H = kc.interactive_adjustment(frame)
            
            # Actualizar court_info en data
            data['court'] = [full_keypoints.tolist(), H.tolist()]       

    print("\nIniciando fases de post-procesamiento...")
    pbar = tqdm(total=4, desc="Progreso General")

    # 1. Fase de Extracción de Datos (Jugadores y cortes de vídeo)
    pbar.set_postfix_str("Extracción de Datos y Cortes")
    data = _extract_features(data, H)
    pbar.update(1)
    
    # 2. Fase de Limpieza (Pelota, con conocimiento de los cortes)
    pbar.set_postfix_str("Limpieza de Datos")
    cut_frames = data.get('cut_frames', [])
    data = _clean_data(data, H, video_path, cut_frames)
    
    # 2.5 Aplicar homografía a la pelota ya limpia
    if 'ball' in data and data['ball'] and H is not None:
        ball_df = pd.DataFrame(data['ball'])
        ball_df = apply_homography(ball_df, H, 'center_x', 'center_y')
        data['ball'] = ball_df.to_dict(orient='records')
    
    pbar.update(1)
    
    # 3. Clasificación de Eventos
    pbar.set_postfix_str("Seguimiento de Eventos")
    interpolated_ball_list = data['ball']
    players_history = data['players']
    
    interpolated_ball_dict = {b['frame']: b for b in interpolated_ball_list}
    event_tracker = EventTracker()
    cut_frames = data.get('cut_frames', [])
    event_tracker.track(interpolated_ball_dict, players_history, cut_frames)
    pbar.update(1)
    
    pbar.set_postfix_str("Clasificación de Golpes")
    stroke_classifier = StrokeClassifier()
    events = stroke_classifier.classify_events(event_tracker.get_history(), players_history, interpolated_ball_dict, cut_frames)
    events = event_tracker.get_history()
    
    # Asignar la mano de la pala dominante al diccionario general de cada jugador
    players_hands = stroke_classifier.get_players_racket_hands()
    players_metadata = {}
    for pid, hand in players_hands.items():
        players_metadata[str(pid)] = {'racket_hand': hand}
        
    data['players_metadata'] = players_metadata

    pbar.update(1)
    pbar.close()
            
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
    
def _clean_data(data, H, video_path, cut_frames):
    ball_history = data.get('ball', [])
    
    if ball_history:
        # Aplanar detecciones si existen
        flat_ball_history = []
        for frame_data in ball_history:
            if 'detections' in frame_data:
                detections = frame_data['detections']
                valid_detections = []
                for det in detections:
                    if H is not None:
                        cx = (det['x_min'] + det['x_max']) / 2
                        cy = (det['y_min'] + det['y_max']) / 2
                        pts = np.array([[[cx, cy]]], dtype=np.float32)
                        transformed = cv2.perspectiveTransform(pts, H)
                        real_x, real_y = transformed[0][0]
                        
                        # Filtrar pelotas estáticas en los laterales de la red
                        if abs(real_x) > 4.2 and abs(real_y) < 1.5:
                            continue
                    valid_detections.append(det)

                if valid_detections:
                    for det in valid_detections:
                        flat_ball_history.append({'frame': frame_data['frame'], 'x_min': det['x_min'], 'y_min': det['y_min'], 'x_max': det['x_max'], 'y_max': det['y_max']})
                else:
                    flat_ball_history.append({'frame': frame_data['frame'], 'x_min': None, 'y_min': None, 'x_max': None, 'y_max': None})
            else:
                flat_ball_history.append(frame_data)
                
        ball_df_raw = pd.DataFrame(flat_ball_history)
        if not ball_df_raw.empty:
                # Lógica de caché MOG2
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                cache_file = os.path.join("data", "outputs", "json", "raw_json", f"{video_name}_mog2_cache.json")
                
                if os.path.exists(cache_file):
                    print(f"Cargando caché de MOG2 desde {cache_file}...")
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    ball_df_mog2 = pd.DataFrame(cached_data)
                else:
                    ball_df_mog2 = remove_false_detections_ball(ball_df_raw.copy(), video_path)
                    print(f"Guardando caché de MOG2 en {cache_file}...")
                    records = ball_df_mog2.where(pd.notnull(ball_df_mog2), None).to_dict(orient='records')
                    with open(cache_file, 'w') as f:
                        json.dump(records, f)
                players_history = data.get('players', [])
                refined_list = filter_ball_outliers(ball_df_mog2, cut_frames=cut_frames, players_history=players_history, raw_ball_df=ball_df_raw)
                data['ball'] = refined_list
    return data

def _extract_features(data, H):
    players_history = data.get('players', {})
    
    if not players_history:
        return data
        
    players_history = calculate_players_centers(players_history)
        
    records = []
    for pid, frames in players_history.items():
        for f_id, f_data in frames.items():
            records.append(f_data)
            
    players_df = pd.DataFrame(records)
    players_df = get_ground_contact_point(players_df)
    
    if H is not None:
        players_df = apply_homography(players_df, H, 'contact_x', 'contact_y')
        
    players_df = remove_static_players(players_df, 300.0)
    players_df = remove_false_detections_players(players_df, 5)
    players_df = normalice_keypoints(players_df)
    players_df, cut_frames = reorder_yolo_ids(players_df)
    data['cut_frames'] = cut_frames
        

    #Volver a meter en un dict
    players_df = players_df.replace({np.nan: None})
    players_history = {}
    for player_id, p_df in players_df.groupby('player_id'):
        str_pid = str(player_id)
        record = p_df.to_dict(orient='records')
        players_history[str_pid] = {str(int(row['frame'])): row for row in record}
                
    data['players'] = players_history

    primer_frame = players_df['frame'].min()

    first_frame_players = players_df[players_df['frame'] == primer_frame]
    return data