from PIL import Image
import lab2_function


# 2. To Gray
img = Image.open("to_gray.png")
lab2_function.display_images((img, lab2_function.to_gray(img)),mode='grayscale', title_arr=("RGB", "Gray"))
#lab2_function.to_gray(img).save('Arseniy.png')

#3.1
ref_img = Image.open("ref.png")
based_img = Image.open("based_img.png")
lab2_function.display_images((based_img, lab2_function.ref_color_correction(based_img, (117, 140, 163), (232,224,222))), title_arr=("Srs", "Destination"))


#3.2 Gray world
lab2_function.display_images((img, lab2_function.gray_world(img)),mode='grayscale', title_arr=("RGB", "Gray world"))

#3.3
img_bad_gray = Image.open("image.png")
lab2_function.display_images((img_bad_gray, lab2_function.log_transform(img_bad_gray)),mode='grayscale', title_arr=("Src", "Log transform"))
lab2_function.display_images((img_bad_gray, lab2_function.linear_transform(img_bad_gray)),mode='grayscale', title_arr=("Src", "Linear transform"))


#5.1 Нормализация гистограммы
hist_img = Image.open("hist.png")
lab2_function.display_images((hist_img, lab2_function.norm_img(hist_img)),mode='grayscale', title_arr=("Input", "Normalize"))
lab2_function.hist_compare((hist_img, lab2_function.norm_img(hist_img)))

#5.2 Эквализация
lab2_function.display_images((hist_img, lab2_function.equalize_img(hist_img)),mode='grayscale', title_arr=("Input", "Equalize"))
lab2_function.hist_compare((hist_img, lab2_function.equalize_img(hist_img)), title_arr=("Input", "Equalize"))

hist_rgb = Image.open("hist_rgb.jpg")
lab2_function.display_images((hist_rgb, lab2_function.equalize_img_rgb(hist_rgb)), title_arr=("Input", "Equalize"))
