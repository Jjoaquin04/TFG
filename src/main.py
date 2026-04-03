import config
from core.court.keypoints import KeypointsCourt
from utils import read_image, make_prediction, open_window, close_window
import cv2

def main():
    # ─── Carga imagen y modelo ────────────────────────────────────────────────────
    img, _ ,_ = read_image(config.DEFAULT_IMAGE_PATH)

    result = make_prediction(config.MODEL_PATH, img)
    
    # Extraemos los keypoints del modelo
    kps = result[0].keypoints.xy.cpu().numpy()

    # ─── Creamos el objeto KeypointsCourt y refinamos los puntos ─────────────────
    exact_court = KeypointsCourt()
    exact_court.refine_points(img, kps[0])  # kps[0] ya que yolo te devuelve shape (1, num_kps, 2)

    # Los puntos refinados ahora viven en exact_court.exact_points
    for i, (ex, ey) in enumerate(exact_court.exact_points):
       cv2.circle(img, (int(round(ex)),int(round(ey))),radius=5,color=(0,0,255),thickness=-1)

    open_window("Keypoint refinados", img)
    close_window()
    
if __name__ == "__main__":
    main()
