import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ==========================================
# UTILIDADES Y MÉTODOS COMUNES
# ==========================================

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
def apply_mog2_leak_catcher(df_iter, raw_ball_df):

    if raw_ball_df is None or raw_ball_df.empty:
        return df_iter
        
    valid_raw = raw_ball_df.dropna(subset=['center_x', 'center_y']).copy()
    if valid_raw.empty:
        return df_iter
        
    # Clustering espacial ultra-ligero
    valid_raw['grid_x'] = (valid_raw['center_x'] // 5) * 5
    valid_raw['grid_y'] = (valid_raw['center_y'] // 5) * 5
    valid_raw['cluster_global'] = valid_raw['grid_x'].astype(int).astype(str) + "_" + valid_raw['grid_y'].astype(int).astype(str)
    
    valid_mog2 = df_iter.dropna(subset=['center_x', 'center_y']).copy()
    valid_raw['id_match'] = valid_raw['frame'].astype(str) + "_" + valid_raw['x_min'].astype(str)
    
    if not valid_mog2.empty:
        valid_mog2['id_match'] = valid_mog2['frame'].astype(str) + "_" + valid_mog2['x_min'].astype(str)
        survivors_ids = set(valid_mog2['id_match'])
        
        valid_raw['survived'] = valid_raw['id_match'].isin(survivors_ids)
        
        cluster_stats = valid_raw.groupby('cluster_global').agg(
            total=('survived', 'count'),
            vivos=('survived', 'sum')
        )
        cluster_stats['muertos'] = cluster_stats['total'] - cluster_stats['vivos']
        cluster_stats['tasa_muerte'] = cluster_stats['muertos'] / cluster_stats['total']
        
        # Zonas tóxicas
        toxic_clusters = cluster_stats[(cluster_stats['muertos'] > 15) & (cluster_stats['tasa_muerte'] > 0.8)].index
        
        leaks_df = valid_raw[(valid_raw['cluster_global'].isin(toxic_clusters)) & (valid_raw['survived'] == True)]
        leaks_ids = set(leaks_df['id_match'])
        
        df_iter['id_match'] = df_iter['frame'].astype(str) + "_" + df_iter['x_min'].astype(str)
        leaks_mask = df_iter['id_match'].isin(leaks_ids)
        
        if leaks_mask.any():
            for col in ['x_min', 'y_min', 'x_max', 'y_max', 'center_x', 'center_y']:
                if col in df_iter.columns:
                    df_iter.loc[leaks_mask, col] = None
                
        df_iter = df_iter.drop(columns=['id_match'], errors='ignore')
        
    return df_iter

# ==========================================
# FUNCIONES PRINCIPALES EXPORTADAS
# ==========================================

def apply_tribunal_inferior(threads, df_iter, video_path):
    """
    Fase 2: Purga rápida de mini-hilos (Tribunal Inferior).
    Elimina hilos <= 2 frames, hilos estáticos (< 20px de movimiento), 
    y hilos totalmente fuera del polígono expandido de la pista.
    """
    import numpy as np
    survivors = []
    
    halo_polygon = None
    if video_path is not None:
        import os
        import cv2
        if os.path.exists(video_path):
            from src.core.court.keypoints import KeypointsCourt
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                court = KeypointsCourt()
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
            import cv2
            all_outside = True
            for p in pts:
                if cv2.pointPolygonTest(halo_polygon, (float(p[0]), float(p[1])), False) >= 0:
                    all_outside = False
                    break
            if all_outside:
                continue
                
        # 2. Filtro Estático (Sólo aplicable a hilos que han vivido lo suficiente)
        # Si un hilo dura 1 o 2 frames, no le ha dado tiempo a moverse, le damos el beneficio de la duda.
        # Pero si dura más de 3 frames y apenas se ha movido 20px, es un objeto estático (logo, reflejo).
        if len(pts) > 3:
            dx = np.max(pts_arr[:, 0]) - np.min(pts_arr[:, 0])
            dy = np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1])
            if np.sqrt(dx**2 + dy**2) < 20.0:
                continue
                
        # Los hilos muy cortitos (incluso de 1 frame) que superan el halo, sobreviven
        survivors.append(thread)
        
    return survivors

def link_tracklets_fase3(threads, df_iter):
    """
    Fase 3: Unión de Hilos (Tracklet Linking) para saltar oclusiones.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    
    if len(threads) <= 1:
        return threads
        
    # Extraer info temporal y geométrica de cada hilo
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
                
            dist_espacial = np.linalg.norm(B['start_pos'] - A['end_pos'])
            
            # Regla estricta: No permitir teletransportes absurdos
            if dist_espacial > 150.0:
                continue
            
            # Excepción Rebote
            if dt <= 2 and dist_espacial <= 30.0:
                # Usamos la distancia como coste en lugar de 0, para que si hay varios 
                # ruidos cerca, se quede con el más próximo.
                cost_matrix[i, j] = dist_espacial 
                continue
                
            # Modelo Cinemático
            proj_pos = A['end_pos'] + A['inertia'] * dt
            E_espacial = np.linalg.norm(B['start_pos'] - proj_pos)
            
            # Regla estricta: Si la predicción falla por mucho, NO enlazamos
            if E_espacial > 80.0:
                continue
                
            cost_matrix[i, j] = w1 * E_espacial + w2 * dt
            
    # Resolución Húngara
    cost_matrix_hungarian = cost_matrix.copy()
    cost_matrix_hungarian[cost_matrix_hungarian == np.inf] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost_matrix_hungarian)
    
    # Extraer los emparejamientos válidos
    next_threads = {i: i for i in range(N)}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < np.inf:
            next_threads[r] = c
            
    # Construir súper-hilos
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

def apply_tribunal_supremo(threads, df_iter):
    """
    Fase 4: Tribunal Supremo (Extracción Final).
    Busca la Trayectoria Verdadera usando Longitud e Inercia (Zig-zag ratio).
    """
    import numpy as np
    import pandas as pd
    
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
        
        # Ratio de Inercia (Versión Robusta)
        # En lugar de usar la distancia neta (Fin - Inicio), usamos la "Diagonal Máxima" (Max Span).
        # Esto salva los globos verticales y los botes de saque, ya que si vuelve al mismo sitio 
        # la distancia neta sería 0, pero el Max Span medirá la altura del globo o del bote.
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
        
    # Selección Greedy: Ordenamos por score descendente
    sorted_indices = np.argsort(scores)[::-1]
    
    valid_tracks = set()
    occupied_frames = {}
    death_causes = {}
    
    for idx in sorted_indices:
        score = scores[idx]
        t = threads[idx]
        
        # Calculamos la amplitud (span) máxima
        t_indices = t['indices_df']
        t_data = df_iter.loc[t_indices]
        t_frames = t_data['frame'].values
        t_centers = t_data[['center_x', 'center_y']].values
        
        min_x, max_x = np.min(t_centers[:, 0]), np.max(t_centers[:, 0])
        min_y, max_y = np.min(t_centers[:, 1]), np.max(t_centers[:, 1])
        max_span = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        
        # Filtro Absoluto: Ruido de menos de 3 puntos
        if score < 3.0:
            death_causes[t['id']] = f"Score {score:.1f}<3"
            continue
            
        # Filtro de Estático Absoluto: Si NUNCA sale de una caja de 80px, es ruido puro disperso (marcador, logo, zapato)
        # Una pelota real (incluso un bote antes de sacar) cubre mucho más de 80 píxeles.
        if max_span < 80.0:
            death_causes[t['id']] = f"Estatico Absoluto ({max_span:.0f}px)"
            continue
            
        # Filtro de Hilos Cortos-Lentos: Si dura poco y encima no se mueve casi nada, es un brazo/hombro de jugador
        if score < 8.0 and max_span < 150.0:
            death_causes[t['id']] = f"Corto-Lento ({score:.1f}, {max_span:.0f}px)"
            continue
            
        is_impostor = False
        overlap_count = 0
        teleport = False
        
        # Comprobar solapamiento temporal y espacial
        for f, (cx, cy) in zip(t_frames, t_centers):
            if f in occupied_frames:
                overlap_count += 1
                min_dist = float('inf')
                for (wx, wy) in occupied_frames[f]:
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array([wx, wy]))
                    if dist < min_dist:
                        min_dist = dist
                
                # Si en el mismo frame ocurren dos hilos pero están a >50px, son objetos físicos distintos.
                if min_dist > 50.0:
                    is_impostor = True
                    break
        
        # Comprobar teletransporte absurdo respecto a ganadores anteriores (relleno de huecos falso)
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
                    # Si requiere moverse a más de 120 px/frame (aprox 3600 px/segundo), es físicamente imposible.
                    # Significa que este hilo es ruido en otra parte de la pista.
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


def link_tracks_cero_memoria(df_iter, R_max=50.0):
    """Fase 1: Enlazado Vectorizado (Cero Memoria)."""
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    grouped_frame = df_iter.groupby('frame')
    thread_alive = []
    thread_finished = []
    next_thread_id = 0

    for frame_idx, frame_data in grouped_frame:
        valid_frame = frame_data.dropna(subset=['center_x', 'center_y'])
        if valid_frame.empty:
            continue
            
        new_detections = valid_frame[['center_x', 'center_y']].to_numpy()
        indices_nuevos = valid_frame.index.to_numpy()

        if len(thread_alive) == 0:
            for i in range(len(new_detections)):
                thread_alive.append({
                    'id': next_thread_id,
                    'positions': [new_detections[i]],
                    'indices_df': [indices_nuevos[i]]
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
        matriz_distancias = cdist(posiciones_esperadas, new_detections)

        # 3. Asignación óptima (Húngaro)
        filas_hilos, columnas_det = linear_sum_assignment(matriz_distancias)

        # 4. Ruptura estricta y nacimientos
        hilos_que_sobreviven = []
        detecciones_usadas = set()

        for idx_hilo, idx_det in zip(filas_hilos, columnas_det):
            distancia = matriz_distancias[idx_hilo, idx_det]
            if distancia <= R_max:
                hilo = thread_alive[idx_hilo]
                hilo['positions'].append(new_detections[idx_det])
                hilo['indices_df'].append(indices_nuevos[idx_det])
                hilos_que_sobreviven.append(hilo)
                detecciones_usadas.add(idx_det)
            else:
                thread_finished.append(thread_alive[idx_hilo])

        hilos_no_emparejados = set(range(len(thread_alive))) - set(filas_hilos)
        for i in hilos_no_emparejados:
            thread_finished.append(thread_alive[i])

        for i in range(len(new_detections)):
            if i not in detecciones_usadas:
                hilos_que_sobreviven.append({
                    'id': next_thread_id,
                    'positions': [new_detections[i]],
                    'indices_df': [indices_nuevos[i]]
                })
                next_thread_id += 1

        thread_alive = hilos_que_sobreviven

    thread_finished.extend(thread_alive)
    return thread_finished

def filter_ball_outliers(ball_history, max_pixels_per_frame=480.0, max_gap_frames=9, max_jump_px=300.0, static_px=20.0, frames_cortes=None, players_history=None, raw_ball_df=None, **kwargs):
    print(f"4. EMPEZANDO FASE DE REFINAMIENTO DE DETECCIONES DE LA PELOTA...\n")

    # Inicialización
    if isinstance(ball_history, pd.DataFrame):
        df_ball = ball_history.copy()
        if df_ball.index.name == 'frame': df_ball = df_ball.reset_index()
    else:
        from src.utils.data_extraction.extractor import calculate_ball_centers
        df_ball = pd.DataFrame(calculate_ball_centers(ball_history))
        if 'frame' not in df_ball.columns and df_ball.index.name == 'frame': df_ball = df_ball.reset_index()
            
    df_iter = ensure_centers(df_ball)
    raw_ball_df = ensure_centers(raw_ball_df)
    
    # MOG2 Leak Catcher
    df_iter = apply_mog2_leak_catcher(df_iter, raw_ball_df)

    # Fase 1: Track Linker Vectorizado (Cero Memoria)
    threads_f1 = link_tracks_cero_memoria(df_iter, R_max=50.0)

    # Fase 2: Tribunal Inferior (Purga Rápida)
    video_path = kwargs.get('video_path')
    threads_f2 = apply_tribunal_inferior(threads_f1, df_iter, video_path)

    # Fase 3: Unión de Hilos (Tracklet Linking)
    threads_f3 = link_tracklets_fase3(threads_f2, df_iter)

    # Fase 4: Tribunal Supremo
    df_full, threads_f4, valid_tracks, death_causes = apply_tribunal_supremo(threads_f3, df_iter)
        
    final_list = interpolate_ball(df_full, max_gap_frames=max_gap_frames, cuts=frames_cortes)
    return final_list

    

def remove_false_detections_ball(ball_df, video_path, mog_history=200, mog_varThreshold=50, mog_ratio=0.17):
    print(f"3. EMPEZANDO FASE DE ELIMINACION DE FALSE DETECTIONS DE LA PELOTA...\n")
    if ball_df.empty: return ball_df

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error abriendo el video para limpiar la pelota: {video_path}")
        return ball_df

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=mog_varThreshold, detectShadows=False)
    valid_dataframes = []
    df_grouped = ball_df.groupby('frame')
    frame_idx = 1
    
    while cap.isOpened():
        if frame_idx not in df_grouped.groups and frame_idx % 30 != 0:
            ret = cap.grab()
            if not ret: break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret: break
            
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        fgmask = bg_subtractor.apply(frame_small)
        
        if frame_idx in df_grouped.groups:
            frame_detections = df_grouped.get_group(frame_idx).copy().dropna(subset=['x_min'])
            if not frame_detections.empty:
                height, width = fgmask.shape[:2]
                frame_detections['x_min_s'] = (frame_detections['x_min'] // 2).clip(lower=0).astype(int)
                frame_detections['y_min_s'] = (frame_detections['y_min'] // 2).clip(lower=0).astype(int)
                frame_detections['x_max_s'] = (frame_detections['x_max'] // 2).clip(upper=width).astype(int)
                frame_detections['y_max_s'] = (frame_detections['y_max'] // 2).clip(upper=height).astype(int)

                valid_indices = []
                for row in frame_detections.itertuples():
                    window = fgmask[row.y_min_s:row.y_max_s, row.x_min_s:row.x_max_s]                    
                    if window.size > 0:
                        window_clean = cv2.medianBlur(window, 5) if window.shape[0] >= 5 and window.shape[1] >= 5 else window 
                        if (cv2.countNonZero(window_clean) / window.size) > mog_ratio:
                            valid_indices.append(row.Index)

                if valid_indices:
                    valid_dataframes.append(frame_detections.loc[valid_indices, ball_df.columns])
                
        frame_idx += 1
    cap.release()
    
    if not valid_dataframes:
        return pd.DataFrame(columns=ball_df.columns)
        
    return pd.concat(valid_dataframes).sort_values('frame').reset_index(drop=True)