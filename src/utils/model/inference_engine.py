from ultralytics import YOLO


def make_prediction_batch(model, imgs, conf_grade: float, batch_size):
    return model.predict(source=imgs, conf=conf_grade, verbose=False, batch=batch_size)

def make_track_batch(model, imgs, batch_size):
    return model.track(source=imgs, persist=True, conf=0.25, verbose=False, batch=batch_size)
