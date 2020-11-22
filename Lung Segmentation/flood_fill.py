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

    image_rgb = cv2.imread("1-039.jpg")
    image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)
    '''
    rows = image.shape[0]
    cols = image.shape[1]
    print("length: ", len(image))
    print("Dimensions: ", image.shape)
    print("rows,cols: ", rows, cols)
    '''
    astranaut = data.astronaut()
    print(astranaut.shape)
    #image_show(astranaut)
    astranaut_gray = cv2.cvtColor(astranaut, cv2.COLOR_BGR2GRAY)
    #image_show(astranaut_gray)
    seed_point = (255,255)
    flood_mask = seg.flood(image,seed_point,tolerance = 0.1)
    fig, ax = image_show(image)
    ax.imshow(flood_mask, alpha=0.3)
    ax.plot(255,255,'bo')



    plt.show()

def circle_points(resolution, center, radius):

    radians = np.linspace(0,2*np.pi,resolution)
    c = center[1] + radius*np.cos(radians)
    r = center[0] + radius*np.sin(radians)

    return np. array([c,r]).T




def image_show(image,nrows=1,ncols=1,cmap='gray',**kwargs):

    fig,ax = plt.subplots(nrows=ncols,ncols=nrows,figsize=(16,16))
    ax.imshow(image,cmap='gray')
    ax.axis("off")
    return fig,ax









if __name__ == "__main__":

    main()