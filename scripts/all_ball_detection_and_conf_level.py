import cv2
import os
import argparse
import pandas as pd
from ultralytics import YOLO
import config
import numpy as np
from pipeline.postprocessing import filter_ball_outliers, interpolate_ball

def main(video_path, output_path=None):
    if output_path is None:
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join('data', 'outputs', 'videos', f"{video_name}_ball_detections.mp4")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Cargando modelo de la pelota desde {config.BALL_MODEL}...")
    model = YOLO(config.BALL_MODEL, task='detect')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error abriendo el video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("\n--- Extrayendo detecciones del video completo ---")
    frame_idx = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, verbose=False)
        
        # Get confidences
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []
        if len(confs) > 0:
            conf_str = ", ".join([f"{c:.2f}" for c in confs])
            print(f"Frame {frame_idx}/{total_frames} - Detecciones: {len(confs)}, Confianzas: [{conf_str}]    ", end='\r')
        else:
            print(f"Frame {frame_idx}/{total_frames} - Sin detecciones", end='\r')

        out.write(results[0].plot())
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"\n¡Renderizado terminado! Video guardado en {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera video de la pelota filtrando outliers')
    parser.add_argument('video', type=str, help='Ruta al video de entrada')
    parser.add_argument('--out', type=str, default=None, help='Ruta de salida del video')
    args = parser.parse_args()
    main(args.video, args.out)
