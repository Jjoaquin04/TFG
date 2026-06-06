import cv2
import os
import argparse
import pandas as pd
from ultralytics import YOLO
import config
from pipeline.postprocessing import filter_ball_outliers

def main(video_path, output_path=None, window_size=7, threshold_dist=50):
    if output_path is None:
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join('data', 'outputs', 'videos', f"{video_name}_filtered_w{window_size}_d{threshold_dist}.mp4")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Cargando modelo de la pelota desde {config.BALL_MODEL}...")
    model = YOLO(config.BALL_MODEL)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error abriendo el video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("\n--- Extrayendo detecciones del video completo ---")
    ball_history = []
    frame_idx = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"Frame {frame_idx}/{total_frames}", end='\r')
        results = model.predict(source=frame, conf=0.40, verbose=False)[0]
        
        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            ball_history.append({
                'frame': frame_idx, 
                'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2, 'conf': conf
            })
        else:
            ball_history.append({
                'frame': frame_idx, 
                'x_min': None, 'y_min': None, 'x_max': None, 'y_max': None, 'conf': None
            })
            
        frame_idx += 1
        
    print("\n\n--- Aplicando filtro de media móvil ---")
    print(f"Parámetros -> Window Size: {window_size} | Threshold Dist: {threshold_dist}")
    
    df_filtered = filter_ball_outliers(ball_history, window_size=window_size, threshold_dist=threshold_dist)
    
    print("\n--- Renderizando video filtrado ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"Renderizando frame {frame_idx}/{total_frames}", end='\r')
        
        if frame_idx in df_filtered.index:
            row = df_filtered.loc[frame_idx]
            
            if pd.notna(row['x_min']):
                x1, y1, x2, y2 = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
                conf = row.get('conf', 0)
            
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if pd.notna(conf):
                    text = f"{conf:.2f}"
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detected = True
            else:
                detected = False
        else:
            detected = False
            
        if not detected:
            cv2.putText(frame, "no deteccion o outlier", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"\n¡Renderizado terminado! Video guardado en {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera video de la pelota filtrando outliers')
    parser.add_argument('video', type=str, help='Ruta al video de entrada')
    parser.add_argument('--out', type=str, default=None, help='Ruta de salida del video')
    parser.add_argument('--window', type=int, default=7, help='Tamano de la ventana de la media movil')
    parser.add_argument('--dist', type=float, default=50.0, help='Umbral de distancia maxima respecto a la media')
    args = parser.parse_args()
    main(args.video, args.out, args.window, args.dist)
