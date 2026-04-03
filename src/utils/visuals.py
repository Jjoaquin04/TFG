import cv2


def draw_line_on_roi(img_roi, a, b, c, color, thickness=1):
    # Dibuja la recta ax+by=c sobre la imagen de la ROI.
    h, w = img_roi.shape[:2]
    pts = []
    if abs(b) > 1e-6:
        pts += [(0,        int(round(c / b))),
                (w, int(round((c - a * w) / b)))]
    if abs(a) > 1e-6:
        pts += [(int(round(c / a)),        0),
                (int(round((c - b * h) / a)), h)]
    if len(pts) >= 2:
        cv2.line(img_roi, pts[0], pts[1], color, thickness, cv2.LINE_AA)

def draw_line_on_global(image, a, b, c, color):
    h, w = image.shape[:2]
    pts = []
    if abs(b) > 1e-6:
        pts += [(0,        int(round(c / b))),
                (w, int(round((c - a * w) / b)))]
    if abs(a) > 1e-6:
        pts += [(int(round(c / a)),        0),
                (int(round((c - b * h) / a)), h)]
    if len(pts) >= 2:
        cv2.line(image, pts[0], pts[1], color, 1, cv2.LINE_AA)