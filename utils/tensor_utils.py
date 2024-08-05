import numpy as np


def pad_to_square(image: np.ndarray, fill: int = 128) -> np.ndarray:
    h, w, c = image.shape
    if h > w:
        length = h - w
        padding = fill * np.ones((h, length, c), dtype=np.uint8)
        left, right = padding[:, :length//2], padding[:, length//2:]
        image = np.concatenate([left, image, right], axis=1)
    elif h < w:
        length = w - h
        padding = fill * np.ones((length, w, c), dtype=np.uint8)
        top, bottom = padding[:length//2], padding[length//2:]
        image = np.concatenate([top, image, bottom], axis=0)
    return image