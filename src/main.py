import cv2 
from ultralytics import YOLO

img = cv2.imread('input_videos/profesional_milano_042.jpg')

model = YOLO('models/yolov8m-keypoints-court.pt')
results = model(source=img,conf=0.25,verbose=False)[0]
annotated_frame = results.plot()
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', annotated_frame)
cv2.waitKey(0)  
cv2.destroyAllWindows()

