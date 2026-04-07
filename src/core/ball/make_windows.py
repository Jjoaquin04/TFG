from collections import deque

import cv2
import numpy as np


class make_windows():
    def __init__(self, dataset):
        self.buffer_frames = deque(maxlen=3)

    def update_frames(self, frame):

        frame = cv2.resize(frame, (640, 360))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        normalized_frame = frame_rgb.astype('float32') / 255.0

        self.buffer_frames.append(normalized_frame)

        if len(self.buffer_frames) == 3: 

            nine_channel = np.concatenate(self.buffer_frames, axis=2)

            tensor = np.transpose(nine_channel, (2, 0, 1))

            return tensor
        else:
            return None