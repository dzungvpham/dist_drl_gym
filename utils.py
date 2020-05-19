import cv2
import numpy as np


def get_preprocessor(h, w, game=None):
    crop_config = {
        "Pong-v0": ((34, 0), (193, 160)),
        "CartPole-v1": ((160, 220), (320, 380))
    }

    if game in crop_config:
        ((x_start, y_start), (x_end, y_end)) = crop_config[game]
        def f(screen):
            screen = screen[x_start:x_end, y_start:y_end, ...]
            screen = cv2.resize(screen, dsize=(h, w), interpolation=cv2.INTER_AREA)
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) / 255.0
            return screen.astype('float32')
        return f
    else:
        def f(screen):
            screen = cv2.resize(screen, dsize=(h, w), interpolation=cv2.INTER_AREA)
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) / 255.0
            return screen.astype('float32')
        return f
