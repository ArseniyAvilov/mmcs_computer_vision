import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def make_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))

def to_fourier(img):
    dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
    return np.fft.fftshift(dft) # преобразование Фурье


def perfect_filter_blur(img, const):
    rows, cols = img.shape #row column
    crow, ccol = int(rows / 2), int(cols / 2) #center
    mask = np.zeros((rows, cols, 2), np.int8) # Создать маску, 2 канала, 256 бит
    mask[crow-const: crow + const, ccol-const: ccol + const] = 1

    md= to_fourier(img) * mask
    ishift=np.fft.ifftshift(md)
    io=cv2.idft(ishift)
    return mask[:,:,0], cv2.magnitude(io[:,:, 0], io[:,:, 1]) # Обратное преобразование Фурье

def perfect_filter_sharpness(img, const):
    rows, cols = img.shape #row column
    crow, ccol = int(rows / 2), int(cols / 2) #center
    mask = np.ones((rows, cols, 2), np.int8) # Создать маску, 2 канала, 256 бит
    mask[crow-const: crow + const, ccol-const: ccol + const] = 0

    md= to_fourier(img) * mask
    ishift=np.fft.ifftshift(md)
    io=cv2.idft(ishift)
    return mask[:,:,0], cv2.magnitude(io[:,:, 0], io[:,:, 1]) # Обратное преобразование Фурье

def butterworth_filter_blur(img, const, n):
    rows, cols = img.shape #row column
    crow, ccol = int(rows / 2), int(cols / 2) #center
    mask1 = [[(abs(ccol - v)**2 + abs(crow - u)**2)**0.5 for v in range(cols)] for u in range(rows)]
    mask = np.ones((rows, cols, 2))
    mask[:,:,0], mask[:,:,1] = mask1, mask1
    mask = 1. / (1 + (mask / const)**(2*n))

    md= to_fourier(img) * mask
    ishift=np.fft.ifftshift(md)
    io=cv2.idft(ishift)
    return mask[:,:,0], cv2.magnitude(io[:,:, 0], io[:,:, 1]) # Обратное преобразование Фурье



def butterworth_filter_sharpness(img, const, n):
    rows, cols = img.shape #row column
    crow, ccol = int(rows / 2), int(cols / 2) #center
    mask1 = [[(abs(ccol - v)**2 + abs(crow - u)**2)**0.5 for v in range(cols)] for u in range(rows)]
    mask = np.ones((rows, cols, 2))
    mask[:,:,0], mask[:,:,1] = mask1, mask1
    mask = 1. / (1 + (const / mask)**(2*n))

    md= to_fourier(img) * mask
    ishift=np.fft.ifftshift(md)
    io=cv2.idft(ishift)
    return mask[:,:,0], cv2.magnitude(io[:,:, 0], io[:,:, 1]) # Обратное преобразование Фурье


def gaussian_filter_blur(img, const):
    rows, cols = img.shape #row column
    crow, ccol = int(rows / 2), int(cols / 2) #center
    mask1 = [[(abs(ccol - v)**2 + abs(crow - u)**2)**0.5 for v in range(cols)] for u in range(rows)]
    mask = np.ones((rows, cols, 2))
    mask[:,:,0], mask[:,:,1] = mask1, mask1
    mask = np.exp((-mask**2) / (2*const*const))

    md= to_fourier(img) * mask
    ishift=np.fft.ifftshift(md)
    io=cv2.idft(ishift)
    return mask[:,:,0], cv2.magnitude(io[:,:, 0], io[:,:, 1]) # Обратное преобразование Фурье

def gaussian_filter_sharpness(img, const):
    rows, cols = img.shape #row column
    crow, ccol = int(rows / 2), int(cols / 2) #center
    mask1 = [[(abs(ccol - v)**2 + abs(crow - u)**2)**0.5 for v in range(cols)] for u in range(rows)]
    mask = np.ones((rows, cols, 2))
    mask[:,:,0], mask[:,:,1] = mask1, mask1
    mask = 1. - np.exp((-mask**2) / (2*const*const))

    md= to_fourier(img) * mask
    ishift=np.fft.ifftshift(md)
    io=cv2.idft(ishift)
    return mask[:,:,0], cv2.magnitude(io[:,:, 0], io[:,:, 1]) # Обратное преобразование Фурье