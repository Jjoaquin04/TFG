import pandas as pd
import numpy as np
from utils.data_extraction.extractor import calculate_ball_centers

def interpolate_ball(ball_df):
    df_copy = ball_df.copy()
    nan_size = group_nan(df_copy)
    df_ball_interp = ball_df.interpolate(method='linear', limit_direction='both')
    is_big = nan_size > 8

    df_ball_interp.loc[is_big, ['x_min', 'y_min', 'x_max', 'y_max','center_x', 'center_y']] = None
    return df_ball_interp.reset_index().to_dict(orient='records')

def remove_static_players(all_players_df: pd.DataFrame, threshold_variance):
    variances = all_players_df.groupby('player_id')[['center_x', 'center_y']].var()
    total_variance = variances['center_x'] + variances['center_y']
    valids_ids = total_variance[total_variance > threshold_variance].index

    return all_players_df[all_players_df['player_id'].isin(valids_ids)]

def remove_false_detections(all_players_df, threshold_num_frames):
    total_frames = all_players_df.groupby('player_id')['frame'].count()
    valids_ids = total_frames[total_frames > threshold_num_frames].index

    return all_players_df[all_players_df['player_id'].isin(valids_ids)]

def group_nan(ball_frame: pd.DataFrame):
    nan_groups= ball_frame['x_min'].notna().cumsum()
    solo_nans = nan_groups[ball_frame['x_min'].isna()]
    len_consecutive_nan = solo_nans.groupby(solo_nans).size()
    
    len_map = nan_groups.map(len_consecutive_nan).fillna(0)
    return len_map

def filter_ball_outliers(ball_history, window_size=8, threshold_dist=50.0):
    
    df_ball = calculate_ball_centers(ball_history)
    df_copy = df_ball.copy()

    median_x = df_copy['center_x'].rolling(window=window_size, center=True, min_periods=1).median()
    median_y = df_copy['center_y'].rolling(window=window_size, center=True, min_periods=1).median()

    deviation = np.sqrt(((median_x - df_copy['center_x'])**2) + ((median_y - df_copy['center_y'])**2))
    atipic = deviation > threshold_dist
    df_copy.loc[atipic, ['x_min', 'y_min', 'x_max', 'y_max','center_x', 'center_y']] = None

    return df_copy
