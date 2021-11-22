import lab3_function
from PIL import Image
import numpy as np
import cv2



image = cv2.imread('hist.png', 0)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
lab3_function.display_images((image, lab3_function.quantization(image, levels=5)), title_arr=("RGB", "Gray"))


img_ship = cv2.imread('ship.jpg', 0)
lab3_function.display_images((img_ship, lab3_function.global_otsu(img_ship)), mode='grayscale',  title_arr=("RGB", "Gray"))


lab3_function.display_images((image, lab3_function.local_otsu(image, 3)), mode='grayscale',  title_arr=("RGB", "Gray"))

lab3_function.display_images((img_ship, lab3_function.hierarchical_otsu(img_ship, 3)), mode='grayscale',  title_arr=("RGB", "Gray"))
