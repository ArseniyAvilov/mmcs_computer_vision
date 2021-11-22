import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

result_path = "results/"

def make_ring(n, thickness = 2):
    small_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n - thickness, n - thickness))
    big_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    res = np.array(big_ring)
    step = int(thickness/2)
    for i in range(n-thickness):
        for j in range(n-thickness):
            res[i+step][j+step] = small_ring[i][j] ^ big_ring[i+step][j+step]
    return  res

def save_image(image, name):
    pilImage = Image.fromarray(image)
    pilImage.save(result_path + name)


img = cv2.imread('Шестеренки.png', 0)

# 1.B1 = B - hole_ring
hole_size = 97
hole_ring = make_ring(hole_size, 2)
erosion = cv2.erode(img, hole_ring, iterations=1)
plt.imshow(erosion)
plt.show()
save_image(erosion, '001.jpg')

# 2. B2 = B1 + haole_mask
hole_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_size, hole_size))
dilation = cv2.dilate(erosion, hole_mask)
plt.imshow(dilation)
plt.show()
save_image(dilation, '002.jpg')

# 3. B3 = B or B2
image3 = cv2.bitwise_or(img, dilation)
plt.imshow(image3)
plt.show()
save_image(image3, '003.jpg')

# 4. 
gear_size = 280
gear_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gear_size, gear_size))
image4 = cv2.morphologyEx(image3, cv2.MORPH_OPEN, gear_body)
plt.imshow(image4)
plt.show()


ring_spacer_size = 11
sampling_ring_spacer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_spacer_size, ring_spacer_size))
image5 = cv2.dilate(image4, sampling_ring_spacer)
plt.imshow(image5)
plt.show()

ring_width_size = 23
sampling_ring_width= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width_size, ring_width_size))
image6 = cv2.dilate(image5, sampling_ring_width)
plt.imshow(image6)
plt.show()

image7 = cv2.bitwise_xor(image5, image6)
plt.imshow(image7)
plt.show()
save_image(image7, '004.jpg')

# 5. B8 = B and B7
image8 = cv2.bitwise_and(img, image7)
plt.imshow(image8)
plt.show()
save_image(image8, '005.jpg')

# 6. B9 = B8 + tip_spacing
tip_spacing_size = 23
tip_spacing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tip_spacing_size, tip_spacing_size))
image9 = cv2.dilate(image8, tip_spacing)
plt.imshow(image9)
plt.show()
save_image(image9, '006.jpg')

# 10.
image10 = cv2.subtract(image7, image9)
defect_cue_size = 35
defect_cue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (defect_cue_size, defect_cue_size))
image10 = cv2.dilate(image10, defect_cue)
result = cv2.bitwise_or(image10 ,image9)
plt.imshow(result)
plt.show()
save_image(result, 'finale.jpg')
