import cv2
from ultralytics import YOLO


img = cv2.imread("data/inputs/images/frame_000012.png")
model = YOLO("models/yolov8-ball-bbx.pt")

result = model.predict(source=img, conf=0.25, verbose=False)[0]
print(result.boxes)
cv2.imshow("ball detection", result.plot())
cv2.waitKey(0)
cv2.destroyAllWindows()