import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io,color,measure,draw,filters,exposure
from skimage.segmentation import clear_border
import skimage.segmentation as seg
import skimage.data as data

def main():

    image_rgb = cv2.imread("1-039 (2).jpg")
    image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)
    '''
    rows = image.shape[0]
    cols = image.shape[1]
    print("length: ", len(image))
    print("Dimensions: ", image.shape)
    print("rows,cols: ", rows, cols)
    '''
    text = data.page()
    print(text.shape)
    image_show(text)
    fig,ax = plt.subplots(1,1)
    ax.hist(text.ravel(),bins=256,range=[0,255])
    ax.set_xlim(0,256)
    ####Basic threshold
    #segmented = text < 60
    #image_show(segmented)
    ####Thresholding methods
    #text_threshold = filters.threshold_sauvola(text)
    #image_show(text < text_threshold)


    plt.show()





def image_show(image,nrows=1,ncols=1,cmap='gray',**kwargs):

    fig,ax = plt.subplots(nrows=ncols,ncols=nrows,figsize=(16,16))
    ax.imshow(image,cmap='gray')
    ax.axis("off")
    return fig,ax









if __name__ == "__main__":

    main()