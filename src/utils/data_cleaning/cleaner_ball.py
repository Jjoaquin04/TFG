import os
import pandas as pd
import numpy as np
import cv2
import core.court.keypoints
import utils.data_extraction.extractor
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def ensure_centers(df):
    #Comprobar que el dataframe tiene el center_x y center_y 
    if df is not None and not df.empty:
        df_copy = df.copy()
        if 'center_x' not in df_copy.columns:
            df_copy['center_x'] = (df_copy['x_min'] + df_copy['x_max']) / 2
        if 'center_y' not in df_copy.columns:
            df_copy['center_y'] = (df_copy['y_min'] + df_copy['y_max']) / 2
        return df_copy
    return df

def group_nan(ball_frame: pd.DataFrame):
    nan_groups = ball_frame['x_min'].notna().cumsum()
    solo_nans = nan_groups[ball_frame['x_min'].isna()]
    len_consecutive_nan = solo_nans.groupby(solo_nans).size()
    len_map = nan_groups.map(len_consecutive_nan).fillna(0)
    return len_map

def interpolate_ball(ball_df, max_gap_frames=15, cuts=None):
    if cuts is None:
        cuts = []
    df_copy = ball_df.copy()
    nan_size = group_nan(df_copy)
    df_ball_interp = ball_df.interpolate(method='linear', limit_direction='both')
    is_big = nan_size > max_gap_frames

    # Identificar huecos que cruzan un corte
    if cuts:
        nan_groups = df_copy['x_min'].notna().cumsum()
        solo_nans = nan_groups[df_copy['x_min'].isna()]
        len_consecutive_nan = solo_nans.groupby(solo_nans).size()
        for group_id, size in len_consecutive_nan.items():
            if size > 0:
                gap_frames = df_copy.loc[nan_groups == group_id, 'frame']
                if not gap_frames.empty:
                    min_f = gap_frames.min()
                    max_f = gap_frames.max()
                    has_cut = any(min_f <= cut <= max_f for cut in cuts)
                    if has_cut:
                        is_big = is_big | (nan_groups == group_id)

    df_ball_interp.loc[is_big, ['x_min', 'y_min', 'x_max', 'y_max','center_x', 'center_y']] = None
    return df_ball_interp.reset_index().to_dict(orient='records')

# ==========================================
# CAPAS DE FILTRADO Y REFINAMIENTO (PIPELINE)
# ==========================================
def apply_spatial_density_filter(df_iter, grid_size=10, max_detections=45):
    if df_iter is None or df_iter.empty:
        return df_iter
        
    valid_df = df_iter.dropna(subset=['center_x', 'center_y']).copy()
    if valid_df.empty:
        return df_iter
        
    # Asignar a grid
    valid_df['grid_x'] = (valid_df['center_x'] // grid_size)
    valid_df['grid_y'] = (valid_df['center_y'] // grid_size)
    valid_df['grid_id'] = valid_df['grid_x'].astype(int).astype(str) + "_" + valid_df['grid_y'].astype(int).astype(str)
    
    # Contar detecciones por celda
    grid_counts = valid_df['grid_id'].value_counts()
    
    # Identificar celdas con ruido estático
    static_grids = grid_counts[grid_counts > max_detections].index
    
    # Filtrar
    mask_static = valid_df['grid_id'].isin(static_grids)
    leaks_indices = valid_df[mask_static].index
    
    if len(leaks_indices) > 0:
        for col in ['x_min', 'y_min', 'x_max', 'y_max', 'center_x', 'center_y']:
            if col in df_iter.columns:
                df_iter.loc[leaks_indices, col] = None
            
    return df_iter

# ==========================================
# FUNCIONES PRINCIPALES EXPORTADAS
# ==========================================

def apply_initial_filter(threads, df_iter, video_path):
    """
    Fase 2: Purga rápida de mini-hilos (Tribunal Inferior).
    Elimina hilos <= 2 frames, hilos estáticos (< 20px de movimiento), 
    y hilos totalmente fuera del polígono expandido de la pista.
    """
    survivors = []
    
    halo_polygon = None
    if video_path is not None:
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                court = core.court.keypoints.KeypointsCourt()
                court.get_delimited_court(frame)
                if len(court.keypoints) >= 4:
                    pts = np.array(court.keypoints[0:4], dtype=np.int32)
                    rect = cv2.boundingRect(pts)
                    x, y, w, h = rect
                    # Polígono dilatado (Halo)
                    halo_polygon = np.array([
                        [max(0, x-200), 0],
                        [min(frame.shape[1], x+w+200), 0],
                        [min(frame.shape[1], x+w+200), min(frame.shape[0], y+h+200)],
                        [max(0, x-200), min(frame.shape[0], y+h+200)]
                    ], dtype=np.int32)
                    
    for thread in threads:
        pts = thread['positions']
        pts_arr = np.array(pts)
            
        # 1. Filtro Espacial (Halo)
        if halo_polygon is not None:
            all_outside = True
            for p in pts:
                if cv2.pointPolygonTest(halo_polygon, (float(p[0]), float(p[1])), False) >= 0:
                    all_outside = False
                    break
            if all_outside:
                continue
                
        # 2. Filtro Estático (Sólo aplicable a hilos que han vivido lo suficiente)
        # Sdura más de 3 frames y apenas se ha movido 20px, es un objeto estático (logo, reflejo).
        if len(pts) > 3:
            dx = np.max(pts_arr[:, 0]) - np.min(pts_arr[:, 0])
            dy = np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1])
            if np.sqrt(dx**2 + dy**2) < 20.0:
                continue
                
        # Los hilos muy cortitos (incluso de 1 frame) que superan el halo, sobreviven
        survivors.append(thread)
        
    return survivors

def link_tracklets_phase3(threads, df_iter):
    """
    Fase 3: Unión de Hilos (Tracklet Linking) para saltar oclusiones.
    """
    
    if len(threads) <= 1:
        return threads
        
    # Extraer info temporal y geométrica de cada thread
    for t in threads:
        subset = df_iter.loc[t['indices_df']]
        t['start_frame'] = subset['frame'].min()
        t['end_frame'] = subset['frame'].max()
        t['start_pos'] = t['positions'][0]
        t['end_pos'] = t['positions'][-1]
        
        # Vector inercia
        if len(t['positions']) >= 2:
            t['inertia'] = t['positions'][-1] - t['positions'][-2]
        else:
            t['inertia'] = np.array([0.0, 0.0])
            
    N = len(threads)
    cost_matrix = np.full((N, N), np.inf)
    w1, w2 = 1.0, 10.0
    
    # Calcular penalizaciones
    for i in range(N):
        for j in range(N):
            if i == j: continue
            
            A = threads[i]
            B = threads[j]
            dt = B['start_frame'] - A['end_frame']
            
            if dt <= 0 or dt > 5:
                continue
                
            spatial_dist = np.linalg.norm(B['start_pos'] - A['end_pos'])
            
            # Descartar saltos espaciales inviables
            if spatial_dist > 150.0:
                continue
            
            # Enlace de rebote
            if dt <= 2 and spatial_dist <= 30.0:
                cost_matrix[i, j] = spatial_dist 
                continue
                
            # Modelo Cinemático
            proj_pos = A['end_pos'] + A['inertia'] * dt
            spatial_error = np.linalg.norm(B['start_pos'] - proj_pos)
            
            if spatial_error > 80.0:
                continue
                
            cost_matrix[i, j] = w1 * spatial_error + w2 * dt
            
    cost_matrix_hungarian = cost_matrix.copy()
    cost_matrix_hungarian[cost_matrix_hungarian == np.inf] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost_matrix_hungarian)
    
    next_threads = {i: i for i in range(N)}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < np.inf:
            next_threads[r] = c
            
    visited = set()
    super_threads = []
    
    for i in range(N):
        if i in visited: continue
        
        curr = i
        chain = []
        while curr not in visited:
            visited.add(curr)
            chain.append(curr)
            nxt = next_threads.get(curr, curr)
            if nxt == curr: break
            curr = nxt
            
        super_t = {'id': threads[chain[0]]['id'], 'positions': [], 'indices_df': []}
        for idx in chain:
            super_t['positions'].extend(threads[idx]['positions'])
            super_t['indices_df'].extend(threads[idx]['indices_df'])
        super_threads.append(super_t)
        
    return super_threads

def apply_final_filter(threads, df_iter):
    if not threads:
        return df_iter.copy(), threads, set()
        
    scores = []
    for t in threads:
        pts = np.array(t['positions'])
        if len(pts) < 2:
            scores.append(0.0)
            continue
            
        frames = df_iter.loc[t['indices_df'], 'frame'].values
        total_frames = frames.max() - frames.min() + 1
        
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        max_span = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        
        diffs = np.diff(pts, axis=0)
        total_dist = np.sum(np.linalg.norm(diffs, axis=1))
        
        ratio_inercia = max_span / total_dist if total_dist > 0 else 0.0
        score = total_frames * ratio_inercia
        scores.append(score)
        
    if not scores:
        return df_iter.copy(), threads, set()
        
    sorted_indices = np.argsort(scores)[::-1]
    
    valid_tracks = set()
    occupied_frames = {}
    death_causes = {}
    
    for idx in sorted_indices:
        score = scores[idx]
        t = threads[idx]
        
        t_indices = t['indices_df']
        t_data = df_iter.loc[t_indices]
        t_frames = t_data['frame'].values
        t_centers = t_data[['center_x', 'center_y']].values
        
        min_x, max_x = np.min(t_centers[:, 0]), np.max(t_centers[:, 0])
        min_y, max_y = np.min(t_centers[:, 1]), np.max(t_centers[:, 1])
        max_span = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        
        if score < 3.0:
            death_causes[t['id']] = f"Score {score:.1f}<3"
            continue
            
        if max_span < 80.0:
            death_causes[t['id']] = f"Estatico Absoluto ({max_span:.0f}px)"
            continue
            
        if score < 8.0 and max_span < 150.0:
            death_causes[t['id']] = f"Corto-Lento ({score:.1f}, {max_span:.0f}px)"
            continue
            
        is_impostor = False
        overlap_count = 0
        teleport = False
        
        for f, (cx, cy) in zip(t_frames, t_centers):
            if f in occupied_frames:
                overlap_count += 1
                min_dist = float('inf')
                for (wx, wy) in occupied_frames[f]:
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array([wx, wy]))
                    if dist < min_dist:
                        min_dist = dist
                
                if min_dist > 50.0:
                    is_impostor = True
                    break
        
        # Validar saltos temporales
        if not is_impostor and overlap_count == 0:
            t_start = t_frames.min()
            past_frames = [f for f in occupied_frames.keys() if f < t_start]
            if past_frames:
                closest_past = max(past_frames)
                dt = t_start - closest_past
                # Si el hueco es de menos de 30 frames (1 segundo), medimos la velocidad requerida
                if dt < 30:
                    start_pos = t_centers[0]
                    past_pos = np.array(occupied_frames[closest_past][0])
                    dist = np.linalg.norm(start_pos - past_pos)
                    speed = dist / dt
                    # Descartar trayectorias demasiado rápidas para ser reales
                    # Significa que este thread es ruido en otra parte de la pista.
                    if speed > 120.0:
                        teleport = True
                        
        if teleport:
            death_causes[t['id']] = f"Teletransporte ({speed:.0f}px/f)"
        elif not is_impostor:
            valid_tracks.add(t['id'])
            for f, (cx, cy) in zip(t_frames, t_centers):
                if f not in occupied_frames:
                    occupied_frames[f] = []
                occupied_frames[f].append((cx, cy))
                
            if overlap_count == 0:
                death_causes[t['id']] = f"WIN ({score:.1f})"
            else:
                death_causes[t['id']] = f"CLON WIN ({score:.1f})"
        else:
            death_causes[t['id']] = f"Solapado ({score:.1f})"
            
    # Extraer los datos de los hilos ganadores
    valid_indices = []
    for t in threads:
        if t['id'] in valid_tracks:
            valid_indices.extend(t['indices_df'])
            
    df_valid = df_iter.loc[valid_indices].copy()
    
    if not df_valid.empty:
        df_valid = df_valid.drop_duplicates(subset=['frame'])
        min_f, max_f = int(df_iter['frame'].min()), int(df_iter['frame'].max())
        all_frames = pd.DataFrame({'frame': range(min_f, max_f + 1)})
        df_full = pd.merge(all_frames, df_valid, on='frame', how='left')
    else:
        df_full = df_iter.copy()
        
    return df_full, threads, valid_tracks, death_causes


def link_tracks_fast(df_iter, R_max=50.0):
    """Fase 1: Enlazado Vectorizado (Cero Memoria)."""

    grouped_frame = df_iter.groupby('frame')
    thread_alive = []
    thread_finished = []
    next_thread_id = 0

    for frame_idx, frame_data in grouped_frame:
        valid_frame = frame_data.dropna(subset=['center_x', 'center_y'])
        if valid_frame.empty:
            continue
            
        new_detections = valid_frame[['center_x', 'center_y']].to_numpy()
        new_indices = valid_frame.index.to_numpy()

        if len(thread_alive) == 0:
            for i in range(len(new_detections)):
                thread_alive.append({
                    'id': next_thread_id,
                    'positions': [new_detections[i]],
                    'indices_df': [new_indices[i]]
                })
                next_thread_id += 1
            continue

        # 1. Predicción cinemática
        posiciones_esperadas = np.zeros((len(thread_alive), 2))
        for i, thread in enumerate(thread_alive):
            if len(thread['positions']) == 1:
                posiciones_esperadas[i] = thread['positions'][-1]
            else:
                current_position = thread['positions'][-1]
                previous_position = thread['positions'][-2]
                direction_vector = current_position - previous_position
                posiciones_esperadas[i] = current_position + direction_vector

        # 2. Matriz de distancias
        dist_matrix = cdist(posiciones_esperadas, new_detections)

        # 3. Asignación óptima (Húngaro)
        thread_rows, columnas_det = linear_sum_assignment(dist_matrix)

        # 4. Ruptura estricta y nacimientos
        surviving_threads = []
        detecciones_usadas = set()

        for thread_idx, idx_det in zip(thread_rows, columnas_det):
            distancia = dist_matrix[thread_idx, idx_det]
            if distancia <= R_max:
                thread = thread_alive[thread_idx]
                thread['positions'].append(new_detections[idx_det])
                thread['indices_df'].append(new_indices[idx_det])
                surviving_threads.append(thread)
                detecciones_usadas.add(idx_det)
            else:
                thread_finished.append(thread_alive[thread_idx])

        unmatched_threads = set(range(len(thread_alive))) - set(thread_rows)
        for i in unmatched_threads:
            thread_finished.append(thread_alive[i])

        for i in range(len(new_detections)):
            if i not in detecciones_usadas:
                surviving_threads.append({
                    'id': next_thread_id,
                    'positions': [new_detections[i]],
                    'indices_df': [new_indices[i]]
                })
                next_thread_id += 1

        thread_alive = surviving_threads

    thread_finished.extend(thread_alive)
    return thread_finished

def filter_ball_outliers(ball_history, max_pixels_per_frame=480.0, max_gap_frames=9, max_jump_px=300.0, static_px=20.0, cut_frames=None, players_history=None, **kwargs):

    # Inicialización
    if isinstance(ball_history, pd.DataFrame):
        df_ball = ball_history.copy()
        if df_ball.index.name == 'frame': df_ball = df_ball.reset_index()
    else:
        df_ball = pd.DataFrame(utils.data_extraction.extractor.calculate_ball_centers(ball_history))
        if 'frame' not in df_ball.columns and df_ball.index.name == 'frame': df_ball = df_ball.reset_index()
            
    df_iter = ensure_centers(df_ball)
    
    # Filtro de Densidad Espacial (Heatmap)
    df_iter = apply_spatial_density_filter(df_iter)

    # Paso 1: Enlazado rápido por distancia
    threads_f1 = link_tracks_fast(df_iter, R_max=50.0)

    # Paso 2: Filtrado rápido de ruido
    video_path = kwargs.get('video_path')
    threads_f2 = apply_initial_filter(threads_f1, df_iter, video_path)

    # Paso 3: Enlazado de tramos
    threads_f3 = link_tracklets_phase3(threads_f2, df_iter)

    # Paso 4: Extracción de trayectoria final
    df_full, threads_f4, valid_tracks, death_causes = apply_final_filter(threads_f3, df_iter)
        
    final_list = interpolate_ball(df_full, max_gap_frames=max_gap_frames, cuts=cut_frames)
    return final_list

    