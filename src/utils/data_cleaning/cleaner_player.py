import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from utils.data_extraction.extractor import calculate_ball_centers

def remove_static_players(all_players_df: pd.DataFrame, threshold_variance):
    variances = all_players_df.groupby('player_id')[['center_x', 'center_y']].var()
    total_variance = variances['center_x'] + variances['center_y']
    valids_ids = set(total_variance[total_variance > threshold_variance].index)
    
    # Añadir a los que, aunque se muevan poco, están dentro de la pista (jugadores reales)
    # Margen interior de 1 metro (Pista: X de -5 a 5, Y de -10 a 10) -> X entre -4 y 4, Y entre -9 y 9
    if 'real_x' in all_players_df.columns and 'real_y' in all_players_df.columns:
        means = all_players_df.groupby('player_id')[['real_x', 'real_y']].mean()
        inside_ids = means[(means['real_x'].abs() < 4.0) & (means['real_y'].abs() < 9.0)].index
        valids_ids.update(inside_ids)

    return all_players_df[all_players_df['player_id'].isin(valids_ids)]

def remove_false_detections_players(all_players_df, threshold_num_frames):
    total_frames = all_players_df.groupby('player_id')['frame'].count()
    valids_ids = total_frames[total_frames > threshold_num_frames].index
            
    return all_players_df[all_players_df['player_id'].isin(valids_ids)]

