import cv2
import numpy as np


def preprocess_screen(screen, h, w):
    screen = cv2.resize(screen, dsize=(h, w), interpolation=cv2.INTER_AREA)
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) / 255.0
    return screen.astype('float32')
