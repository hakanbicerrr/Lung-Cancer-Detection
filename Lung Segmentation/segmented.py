import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io,color,measure
from skimage import morphology
from scipy.ndimage import distance_transform_edt
from skimage import feature
from sklearn.cluster import KMeans
from skimage.segmentation import clear_border
import random as rng
rng.seed(12345)

def main():

    image_rgb = cv2.imread("1-039.jpg")
    image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)
    plt.figure(1)
    plt.imshow(image, cmap="gray")

    #Smoothing image
    median = cv2.medianBlur(image, 7)
    gaus_med = cv2.GaussianBlur(median, (5,5), 0)
    plt.figure(2)
    plt.imshow(gaus_med,cmap="gray")
    #Thresholding
    ret3, binary_image = cv2.threshold(gaus_med, 160, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_image,kernel,iterations=1)
    binary_image = cv2.dilate(eroded,kernel,iterations=1)

    #binary_image = cv2.adaptiveThreshold(gaus_med, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11, 2)
    plt.figure(3)
    plt.imshow(binary_image,cmap="gray")
    #Object removal
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    print(sizes)
    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 950
    # your answer image
    object_removed = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            object_removed[output == i + 1] = 255

    plt.figure(5)
    plt.imshow(object_removed,cmap="gray")
    #Extracted ROI
    segmented_image = cv2.subtract(np.uint8(binary_image),np.uint8(object_removed))
    plt.figure(6)
    plt.imshow(segmented_image,cmap="gray")
    #Moprhological Process



    print(cv2.connectedComponentsWithStats(segmented_image, connectivity=8))

    contours, _ = cv2.findContours(segmented_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)

    cv2.drawContours(image_rgb, contours, -1, (0, 255, 0), 1)
    plt.figure(9)
    plt.imshow(image_rgb)


    '''#Distance Transform
    dist = cv2.distanceTransform(segmented_image, cv2.DIST_L2, 3)
    cv2.normalize(dist,dist,0,1.0,cv2.NORM_MINMAX)
    plt.figure(7)
    plt.imshow(dist,cmap="gray")
    #Threshold distance
    _, dist = cv2.threshold(dist, 0.1, 1.0, cv2.THRESH_BINARY)
    plt.figure(8)
    plt.imshow(dist,cmap="gray")
    #Find Contours
    dist_8u = dist.astype('uint8')
    '''













    '''
    # flood fill to remove mask at borders of the image
    dilated = abs(dilated-255)
    plt.figure(9)
    plt.imshow(dilated, cmap="gray")
    mask = dilated
    h, w = image.shape[:2]
    for row in range(h):
        if mask[row, 0] == 255:
            cv2.floodFill(mask, None, (0, row), 0)
        if mask[row, w - 1] == 255:
            cv2.floodFill(mask, None, (w - 1, row), 0)
    for col in range(w):
        if mask[0, col] == 255:
            cv2.floodFill(mask, None, (col, 0), 0)
        if mask[h - 1, col] == 255:
            cv2.floodFill(mask, None, (col, h - 1), 0)

    # flood fill background to find inner holes
    holes = mask.copy()
    cv2.floodFill(holes, None, (0, 0), 255)

    # invert holes mask, bitwise or with mask to fill in holes
    holes = cv2.bitwise_not(holes)
    mask = cv2.bitwise_or(mask, holes)

    # display masked image
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    masked_img_with_alpha = cv2.merge([image, image, image, mask])

    cv2.imwrite('masked.png', masked_img)
    cv2.imwrite('masked_transparent.png', masked_img_with_alpha)
    '''


    plt.show()

















if __name__ == "__main__":

    main()