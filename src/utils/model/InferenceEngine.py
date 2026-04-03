from ultralytics import YOLO


def make_prediction(path_model, img):
    model = YOLO(path_model)
    return model.predict(source=img, conf=0.25, verbose=False)[0]