import numpy as np
from PIL import Image
import cv2

def gradation_cor(image, c):
    """
    Вернуть градационную коррекцию image с параметром масштаба c
    """
    image2 = image.copy()
    image2 = image2 - image2.min()
    image2 = c * (image2 / image2.max())
    return image2

def standart(image):
    """
    Возвращает стандартизованную форму (без абсолютных значений) image в uint8
    """
    image2 = image.copy()
    image2[image2 < 0.0] = 0.0
    image2[image2 > 255.0] = 255.0
    image2 = image2.astype(np.uint8)
    return image2

result_path = "results/"

def save_image(image, name):
    pilImage = Image.fromarray(image)
    pilImage.save(result_path + name)



# выделение границ -> abs(intensity)
sobel_X = lambda image: np.uint8(np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0)))
sobel_Y = lambda image: np.uint8(np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1)))

# Sobel - объединение (X, Y)
sobel_XY = lambda image: cv2.bitwise_or(sobel_X(image), sobel_Y(image))


def linear_smooth(image, method, width, height):
    """
    На основании method вернуть нужное сглаживание image с параметрами окна [width; height] 
    """
    options = {
        'average': cv2.blur(image, (width, height)),
        'gaussian': cv2.GaussianBlur(image, (width, height), 0)
    }
    return options[method]

gamm = lambda image, c, gamma: cv2.LUT(image, np.array([np.uint8(np.clip(c * ((i / 255.0) ** gamma), 0, 255)) \
                                              for i in np.arange(0, 256)]).astype("uint8"))

