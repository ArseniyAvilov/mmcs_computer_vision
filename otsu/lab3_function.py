import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2

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


def to_gray(rgb_img):
    rgb_img = np.array(rgb_img)
    return 0.3*rgb_img[:,:,0] + 0.59*rgb_img[:,:,1] + 0.11*rgb_img[:,:,2]

def quantization(img=None, levels=None):
    lvls = list(range(1, 256, int(256/levels)))[1:] + [255, ]
    img = np.array(img)
    for lvl in range(1, len(lvls)):
        img[:,:,0] = np.where((img[:,:,0] >= lvls[lvl-1]) & (img[:,:,0] > lvls[lvl]), lvls[lvl]*0.3, img[:,:,0])
        img[:,:,1] = np.where((img[:,:,1] >= lvls[lvl-1]) & (img[:,:,1] > lvls[lvl]), lvls[lvl]*0.22, img[:,:,1])
        img[:,:,2] = np.where((img[:,:,2] >= lvls[lvl-1]) & (img[:,:,2] > lvls[lvl]), lvls[lvl]*0.6, img[:,:,2])
    return img


def threshold_calculation(img, is_normalized=True):
    bins_num = 256
    
    # Get the image histogram
    hist, bin_edges = np.histogram(img, bins=bins_num)
    
    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1

    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    return threshold


def global_otsu(img):
    threshold = threshold_calculation(img)

    img = np.where(img >= threshold, 255, 0)

    return img

def local_otsu(img, crop_count = 2):

    image_copy = img.copy()
    imgheight=img.shape[0]
    M = int(imgheight / crop_count)
    imgwidth=img.shape[1] 
    N = int(imgwidth / crop_count)

    for y in range(0, imgheight, M):
        for x in range(0, imgwidth, N):
            if (imgheight - y) < M or (imgwidth - x) < N:
                break
             
            y1 = y + M
            x1 = x + N
 
            if x1 >= imgwidth and y1 >= imgheight:
                x1 = imgwidth - 1
                y1 = imgheight - 1
                #Crop into patches of size MxN
                tiles = image_copy[y:y+M, x:x+N]
                tiles = global_otsu(tiles)
                image_copy[y:y+M, x:x+N] = tiles
            elif y1 >= imgheight: # when patch height exceeds the image height
                y1 = imgheight - 1
                #Crop into patches of size MxN
                tiles = image_copy[y:y+M, x:x+N]
                tiles = global_otsu(tiles)
                image_copy[y:y+M, x:x+N] = tiles
            elif x1 >= imgwidth: # when patch width exceeds the image width
                x1 = imgwidth - 1
                #Crop into patches of size MxN
                tiles = image_copy[y:y+M, x:x+N]
                tiles = global_otsu(tiles)
                image_copy[y:y+M, x:x+N] = tiles
            else:
                #Crop into patches of size MxN
                tiles = image_copy[y:y+M, x:x+N]
                tiles = global_otsu(tiles)
                image_copy[y:y+M, x:x+N] = tiles
    
    return image_copy

def threshold_for_hierarchical(hist, bin_edges, is_normalized=True):

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1

    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    return threshold

def hierarchical_otsu(img, steps):
    bins_num = 256
    # Get the image histogram
    hist, bin_edges = np.histogram(img, bins=bins_num)

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    thresholds = []
    recur_otsu(hist, bin_edges, steps, thresholds)
    thresholds.sort()
    for lvl in range(len(thresholds)):
        img = np.where((img >= thresholds[lvl-1]) & (img > thresholds[lvl]), thresholds[lvl]*0.3, img)
    return img



def recur_otsu(hist, bin_edges, steps, thresholds):
    if steps <= 0:
        return
    threshold = threshold_for_hierarchical(hist, bin_edges)
    thresholds.append(threshold)
    right_part, left_part = np.array_split(bin_edges, np.where(bin_edges <= threshold)[0])[-1], np.array_split(bin_edges, np.where(bin_edges > threshold)[0])[0]
    
    left_hist, right_hist = hist[len(right_part)-1:], hist[:len(right_part)-2]
    right_part = right_part[1:]
    
    if steps > 0 and ((left_hist.shape[0] + left_part.shape[0]) > 2) and  ((right_hist.shape[0] + right_part.shape[0]) > 2):
        steps -= 1
        return recur_otsu(left_hist, left_part, steps-1, thresholds) , recur_otsu(right_hist, right_part, steps-1, thresholds)
    if steps > 0 and ((left_hist.shape[0] + left_part.shape[0]) > 2) and  ((right_hist.shape[0]  + right_part.shape[0]) <= 2):
        steps -= 1
        return recur_otsu(left_hist, left_part, steps-1, thresholds)
    if steps > 0 and ((left_hist.shape[0] + left_part.shape[0]) <= 2) and  ((right_hist.shape[0]  + right_part.shape[0]) > 2):
        #steps -= 1
        return recur_otsu(right_hist, right_part, steps-1, thresholds)
    return
