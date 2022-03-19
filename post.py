from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import skimage.measure as measure
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb

def importImage(path, target_size=[256,256], flag_multi_class = False, as_gray = True):
    num_image = (len(os.listdir(path)))

    fig, ax = plt.subplots()
    flag_multi_class = False

    for i in range(num_image):
        file_name = os.listdir(path)[i]
        img = io.imread(os.path.join(path, file_name),as_gray = as_gray)

        image = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else img[:,:]

        segments_fz = felzenszwalb(image, scale=1, sigma=0.000001, min_size=1)
        marked = mark_boundaries(image, segments_fz, color= (255, 255, 255), mode='outer')
        iy , ix, _ = np.where( marked == (255, 255, 255) )
        new = np.zeros((image.shape[:2]))
        new[iy,ix] = 255

        ax.imshow(new, cmap=plt.cm.gray)
        # Find contours at a constant value of 0.8
        # contours = measure.find_contours(image, 0.8)

        # for contour in contours:
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig("data/membrane/contour/image_%d"%i)

importImage("data/membrane/results_success")
