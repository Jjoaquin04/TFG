from ultralytics import YOLO


def make_prediction_batch(model, imgs, conf_grade: float):
    return model.predict(source=imgs, conf=conf_grade, verbose=False, batch=16)

def make_track_batch(model, imgs):
    return model.track(source=imgs, persist=True, conf=0.25, verbose=False, batch=16)
