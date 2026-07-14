import cv2
import os
import sys
from ultralytics import YOLO
from core.court.keypoints import KeypointsCourt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    img_path = 'data/inputs/images/hexagon.png'
    if not os.path.exists(img_path):
        print(f"Error: No se encuentra el video en {img_path}")
        return

    model_path = 'models/yolov8-court-keypoint.pt'
    if not os.path.exists(model_path):
        model_path = 'models/yolov8-court-keypoint_openvino_model'

    print(f"Cargando modelo de pose: {model_path}")
    model = YOLO(model_path)

    frame = cv2.imread(img_path)
    results = model.predict(source=frame, verbose=True)

    if results and len(results) > 0:
        kpts = results[0].keypoints
        if kpts is not None and len(kpts) > 0:
            kps_yolo = kpts.xy.cpu().numpy()[0]
            kc = KeypointsCourt()
            
            print("Ejecutando refine_points (visualización de máscaras)...")
            kc.refine_points(frame, kps_yolo)
            print("\nPrueba finalizada.")
        else:
            print("No se encontraron keypoints.")

if __name__ == "__main__":
    main()
