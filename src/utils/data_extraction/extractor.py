import pandas as pd
import numpy as np
import cv2

def calculate_players_centers(dict_players):
    for _, player_data in dict_players.items():
        for record in player_data.values():
            record['center_x'] = (record['x_min'] + record['x_max']) / 2
            record['center_y'] = (record['y_min'] + record['y_max']) / 2
    return dict_players
        
def calculate_ball_centers(data_ball):
    df_ball = pd.DataFrame(data_ball).set_index('frame')
    df_ball['center_x'] = (df_ball['x_min'] + df_ball['x_max']) / 2
    df_ball['center_y'] = (df_ball['y_min'] + df_ball['y_max']) / 2
    return df_ball

def apply_homography(player_df: pd.DataFrame, homography_matrix, x_col, y_col):
    # Máscara para no transformar NaNs
    mask = player_df[x_col].notna() & player_df[y_col].notna()
    
    if not mask.any():
        player_df['real_x'] = np.nan
        player_df['real_y'] = np.nan
        return df

    pts = player_df.loc[mask, [x_col, y_col]].values
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    
    transformed = cv2.perspectiveTransform(pts, homography_matrix)
    
    player_df.loc[mask, 'real_x'] = transformed[:, 0, 0]
    player_df.loc[mask, 'real_y'] = transformed[:, 0, 1]
    
    return player_df

def get_ground_contact_point(player_df: pd.DataFrame):
    """
    Estrategia:
        1. Promedio de ambos tobillos si ambos son válidos
        2. El tobillo disponible si solo hay uno
        3. Centro inferior del BBX como último recurso
    """
    left_ankle_column = player_df['keypoints'].str[15]
    right_ankle_column = player_df['keypoints'].str[16]

    left_ankle_x = left_ankle_column.str[0]
    left_ankle_y = left_ankle_column.str[1]
    left_ankle_conf =left_ankle_column.str[2]

    right_ankle_x = right_ankle_column.str[0]
    right_ankle_y = right_ankle_column.str[1]
    right_ankle_conf = right_ankle_column.str[2]

    #Preparar la imformacion para los np.where
    CONF_THRESHOLD = 0.5
    cond_left_ankle = left_ankle_conf > CONF_THRESHOLD
    cond_right_ankle = right_ankle_conf > CONF_THRESHOLD

    both_x = (left_ankle_x + right_ankle_x) /2
    both_y = (left_ankle_y + right_ankle_y) /2

    bbx_center_x = (player_df['x_min'] + player_df['x_max']) / 2
    bbx_bottom_y = player_df['y_max']

    player_df['contact_x'] = np.where(
        cond_left_ankle & cond_right_ankle, both_x,
        np.where(
            cond_left_ankle, left_ankle_x,
            np.where(
                cond_right_ankle, right_ankle_x, bbx_center_x
            )
        )
    )
    player_df['contact_y'] = np.where(
        cond_left_ankle & cond_right_ankle, both_y,
        np.where(
            cond_left_ankle, left_ankle_y,
            np.where(
                cond_right_ankle, right_ankle_y, bbx_bottom_y
            )
        )
    )

    return player_df

def normalice_keypoints(player_df: pd.DataFrame):
    left_hip = player_df['keypoints'].str[11]
    right_hip = player_df['keypoints'].str[12]

    (left_hip_x, left_hip_y) = left_hip.str[0], left_hip.str[1]
    (right_hip_x, right_hip_y) = right_hip.str[0], right_hip.str[1]
    
    center_hips_x = (left_hip_x + right_hip_x) / 2.0
    center_hips_y = (left_hip_y + right_hip_y) / 2.0
    
    player_df['norm_keypoints'] = [
        [[kp[0] - center_x ,kp[1] - center_y] for kp in kps]
        for kps, center_x, center_y in zip(player_df['keypoints'], center_hips_x, center_hips_y)
    ]

    return player_df

def reorder_yolo_ids(players_df):
    master_info = _obtain_four_ids(players_df)
    first_bottom_footprint = master_info['bottom_footprint']
    role = master_info['master_role']
    
    players_frame_min = players_df.groupby('player_id')['frame'].min()
    first_frame = players_df['frame'].min()
    new_ids = players_frame_min[players_frame_min > first_frame]
    groups_ids = new_ids.groupby(new_ids).groups
    
    for cut_frame, ids in groups_ids.items():
        players_cut = players_df[players_df['frame'] == cut_frame]
        _reasing_ids(players_cut, first_bottom_footprint, role)

    players_df['player_id'].map(role).fillna(players_df['player_id'])
    print(players_df['player_id'])

def _obtain_four_ids(players_df: pd.DataFrame):
    first_frame = players_df['frame'].min()
    players_first_frame = players_df[players_df['frame'] == first_frame]

    ordered_players = players_first_frame.sort_values(by='contact_y', ascending=True)
    [ordered_ids, top_footprint, bottom_footprint] = _order_four_players(ordered_players)
    role = {player_id: rol for rol, player_id in enumerate(ordered_ids)}
    master_info = {
        'master_role': role, 
        'top_footprint': top_footprint,
        'bottom_footprint': bottom_footprint
    }
    return master_info

def _reasing_ids(players_cut, first_bottom_footprint, master_role):
    ordered_players = players_cut.sort_values(by='contact_y', ascending=True)
    [ordered_ids, top_footprint, bottom_footprint] = _order_four_players(ordered_players)

    pair_keep_in_bottom = _compare_footprint(first_bottom_footprint, bottom_footprint)

    new_ids = [id for id in ordered_ids if id not in master_role]

    for id in new_ids:
        position_in_court = ordered_ids.index(id)

        if pair_keep_in_bottom:
            new_rol = position_in_court
        else:
            if position_in_court == 0:
                new_rol = 3
            elif position_in_court == 1:
                new_rol = 2
            elif position_in_court == 2:
                new_rol = 1
            else:
                new_rol = 0

        master_role[id] = new_rol



def _order_four_players(four_players):
    top_pair = four_players.iloc[:2]
    bottom_pair = four_players.iloc[2:4]

    top_footprint = np.median(top_pair['shirt_color_hsv'].tolist(), axis=0).tolist()
    bottom_footprint = np.median(bottom_pair['shirt_color_hsv'].tolist(), axis=0).tolist()
    variance_y_top = abs(top_pair.iloc[0]['contact_y'] - top_pair.iloc[1]['contact_y'])
    variance_y_bottom = abs(bottom_pair.iloc[0]['contact_y'] - bottom_pair.iloc[1]['contact_y'])

    if variance_y_top > variance_y_bottom:
        delta_x_serve_pair = abs(top_pair.iloc[0]['contact_x'] - top_pair.iloc[1]['contact_x'])
        if delta_x_serve_pair <  50:
            top_pair = top_pair.sort_values(by='contact_y', ascending=False)
        else:
            top_pair = top_pair.sort_values(by='contact_x')
            
        bottom_pair = bottom_pair.sort_values(by='contact_x')
    else:
        delta_x_serve_pair = abs(bottom_pair.iloc[0]['contact_x'] - bottom_pair.iloc[1]['contact_x'])
        if delta_x_serve_pair <  50:
            bottom_pair = bottom_pair.sort_values(by='contact_y', ascending=True)
        else:
            bottom_pair = bottom_pair.sort_values(by='contact_x')
            
        top_pair = top_pair.sort_values(by='contact_x')

    return [top_pair['player_id'].to_list() + bottom_pair['player_id'].to_list(), top_footprint, bottom_footprint]

def _compare_footprint(master_footprint, foot_print):
    mh, ms, mv = master_footprint
    h, s, v = foot_print

    dh = min(abs(mh - h), 180 - abs(mh-h))
    ds = abs(ms - s)
    dv = abs(mv - v)

    euclidean_distance = np.linalg.norm([dh, ds, dv])
    return euclidean_distance < 50  
    