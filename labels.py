import os
import skimage.transform as trans
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

def gen_labels(path):
    images = gen_images(path)
    x = input(len(images))

    path = "data/RednBlue/Axial/Label"
    outpath = "test"


    img = io.imread(os.path.join(path, file_name))
    # iy , ix, _ = np.where(img == (255, 0, 0) )
    new = np.zeros((img.shape))
    for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                x,y,z = img[i,j,0],img[i,j,1],img[i,j,2]
                if (x > 140) & (x < 220) & (y < 90) & (z < 90):
                    new[i,j,0] = 255
                elif (x < 90) & (y < 90) & (z > 140) & (z < 220):
                    new[i,j,2] = 255
                

    x = input()
    x = input(np.where(img))
    iy , ix, _ = np.where((img[0] > 130 & img[1] < 90 & img[2] < 90))
    new[iy,ix] = (255,0,0)
    # iy , ix, _ = np.where(img[0] < 90, img[1] < 90, img[2] > 130)
    # new = np.zeros((img.shape))
    # new[iy,ix] = (0,255,0)

    save(new)

def gen_images(path,target_size = (256,256),flag_multi_class=False):
    images = np.array()
    num_image = (len(os.listdir(path)))
    for i in range(num_image):
        x = input(num_image)
        file_name = os.listdir(path)[i]
        img = io.imread(os.path.join(path, file_name))
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        np.append(images, img)
        x = input(images)
        # yield img
    return images



def labels(img):
    img = img[0,:,:,:,0]
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.savefig("data/RednBlue/Test Results/test")
    plt.savefig("test1")



