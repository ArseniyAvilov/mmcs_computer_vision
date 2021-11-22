import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import PIL
sns.set(style="white", palette="bright", color_codes=True)

def display_images(img_list, size=(8, 8), mode='rgb', title_arr=(None,None)):    
    fig=plt.figure(figsize=size, dpi=150)
    
    for i in range(len(img_list)):
        fig.add_subplot(1, 2, i+1)
        if mode=='grayscale':
            plt.imshow(img_list[i], cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img_list[i])
        plt.title(title_arr[i])
        plt.axis('off')

    plt.show()

def hist_compare(img_list, title_arr=('Input', 'Normalize')):
    _, axes = plt.subplots(1, 2, figsize=(17, 4), sharey=True)
    for i in range(2):
        sns.distplot(img_list[i], color="black", ax=axes[i]).set_xlim(0, 255)
        plt.title(title_arr[i])
    plt.show()

def norm_img(img, cnst=255.0):
    img = np.array(img)
    return cnst*(img-img.min()) / (img.max() - img.min())

def to_gray(rgb_img):
    rgb_img = np.array(rgb_img)
    return 0.3*rgb_img[:,:,0] + 0.59*rgb_img[:,:,1] + 0.11*rgb_img[:,:,2]

def ref_color_correction(main_img, orig_color, ref_color):
    main_img = np.array(main_img)

    R = ref_color[0] / orig_color[0]
    G = ref_color[1] / orig_color[1]
    B = ref_color[2] / orig_color[2]
    main_img[:,:,0] = np.where(main_img[:,:,0]*R > 255, 255, main_img[:,:,0]*R)
    main_img[:,:,1] = np.where(main_img[:,:,1]*G > 255, 255, main_img[:,:,1]*G)
    main_img[:,:,2] = np.where(main_img[:,:,2]*B > 255, 255, main_img[:,:,2]*B)
    return  main_img

def gray_world(img):
    img = np.array(img)
    R = np.mean(img[:, :, 0])
    G = np.mean(img[:, :, 1])
    B = np.mean(img[:, :, 2])
    Avg = (R + G + B) / 3

    img[:, :, 0] = img[:, :, 0]* (Avg / R)
    img[:, :, 1] = img[:, :, 1]* (Avg / G)
    img[:, :, 2] = img[:, :, 2]* (Avg / B) 
    return img

def log_transform(img):
    img = np.array(img)
    c = 255 / np.log(1 + np.max(img)) 
    return (c * (np.log(img + 1))).astype('int')

def linear_transform(img):
    img = np.array(img)
    c = 255 / (np.max(img) - np.min(img))
    return (c * (img - np.min(img))).astype('int')

def norm_image_rgb(img, mul_const=255.0):
    img = np.array(img)
    img = np.zeros(img.shape, dtype='float')
    for i in range(img.shape[2]):
        img[:,:,i] = norm_img(img[:,:,i], mul_const) 
    return img.astype('int')

def equalize_img(img):
    img = np.array(img)
    histogram, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    
    histogram = histogram / (img.shape[0] * img.shape[1])
    
    for i in range(1,256):
        histogram[i] = histogram[i-1] + histogram[i]

    return 255.0 * histogram[img.astype('int')]


def equalize_img_rgb(img):
    img = np.array(img)
    for i in range(img.shape[2]):
        img[:,:,i] = equalize_img(img[:,:,i])
    return img.astype('int')
