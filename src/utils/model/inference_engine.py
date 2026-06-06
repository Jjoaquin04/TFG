from ultralytics import YOLO


def make_prediction(model, img, conf_grade: float):
    return model.predict(source=img, conf=conf_grade, verbose=False)[0]

def make_track(model, img):
    return model.track(source=img, persist=True, conf=0.25, verbose=False)[0]
