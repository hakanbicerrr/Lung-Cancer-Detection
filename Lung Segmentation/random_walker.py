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
    new_image = np.zeros((512,512),np.uint8)
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
    image_labels = np.zeros(image.shape, dtype=np.uint8)
    points = circle_points(110, [255, 255], 180)[:-1]
    indices = draw.circle_perimeter(255,255,50)
    image_labels[indices] = 1
    image_labels[points[:,1].astype(np.int),points[:,0].astype(np.int)] = 2
    image_show(image_labels)
    image_segmented = seg.random_walker(image,image_labels,beta=2500)

    fig,ax = image_show(image)
    ax.imshow(image_segmented == 1,alpha=0.3)

    '''astranaut_labels = np.zeros(astranaut_gray.shape, dtype=np.uint8)
    points = circle_points(200, [100, 220], 100)[:-1]
    indices = draw.circle_perimeter(100, 220, 25)
    astranaut_labels[indices] = 1
    astranaut_labels[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2

    astranaut_segmented = seg.random_walker(astranaut_gray, astranaut_labels, beta=9001)

    fig, ax = image_show(astranaut_gray)
    ax.imshow(astranaut_segmented == 1, alpha=0.3)
    '''
    plt.show()
    new_image[image_segmented==1] = image[image_segmented==1]
    cv2.imshow("seg",new_image)
    cv2.waitKey()
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