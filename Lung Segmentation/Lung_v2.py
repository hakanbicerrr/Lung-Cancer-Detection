import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io,color,measure
from skimage.segmentation import clear_border
from skimage import segmentation
def main():

    image_rgb = cv2.imread("1-039 (2).jpg")
    image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)

    rows = image.shape[0]
    cols = image.shape[1]
    print("length: ",len(image))
    print("Dimensions: ",image.shape)
    print("rows,cols: ",rows,cols)
    #average_image = cv2.blur(image,(3,3))
    #gaussian_image = cv2.GaussianBlur(image,(3,3),0)
    #median_image = cv2.medianBlur(image,3)

    kernel = np.ones((7, 7), np.float32) / 49
    median_2d_5 = cv2.filter2D(image, -1, kernel)

    gaussian_of_median_2d_5 = cv2.GaussianBlur(median_2d_5,(7,7),0)

    (thresh, binary_image) = cv2.threshold(gaussian_of_median_2d_5, 165, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    #img_erosion = cv2.erode(binary_image, kernel, iterations=2)
    #img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)
    opening = cv2.morphologyEx(binary_image,cv2.MORPH_OPEN,kernel,iterations=1)
    #img_erosion = cv2.erode(binary_image, kernel, iterations=2)
    #opening = cv2.dilate(img_erosion, kernel, iterations=2)
    #opening = img_dilation
    plt.figure(1)
    plt.subplot(3, 3, 5), plt.imshow(opening, cmap='gray'), plt.title("Image")
    plt.xticks([]), plt.yticks([])
    opening = clear_border(opening)
    plt.subplot(3, 3, 6), plt.imshow(opening, cmap='gray'), plt.title("Image")
    plt.xticks([]), plt.yticks([])
    cv2.imshow("opening",opening)
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    cv2.imshow("sure_bg",sure_bg)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    #cv2.imshow("dist",dist_transform)
    ret2, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    #cv2.imshow("sure,fg",sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    #cv2.imshow("unknown",unknown)
    ret3,markers = cv2.connectedComponents(sure_fg)
    markers = markers+10
    markers[unknown==255] = 0
    markers = cv2.watershed(image_rgb,markers)
    #markers = segmentation.watershed(image,markers)
    print(markers)
    image_rgb[markers == -1] = [255,0,255]
    img3 = color.label2rgb(markers,bg_label=0)
    plt.figure(2)
    plt.subplot(1, 2, 1), plt.imshow(image_rgb, cmap='gray'), plt.title("Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img3, cmap='gray'), plt.title("Image")
    plt.xticks([]), plt.yticks([])


    plt.figure(1)
    plt.subplot(3,3,1), plt.imshow(image,cmap='gray'), plt.title("Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 2), plt.imshow(median_2d_5, cmap='gray'), plt.title("Median Filtered Image 5x5")
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 3), plt.imshow(gaussian_of_median_2d_5, cmap='gray'), plt.title("Gaussian of Median Image 5x5")
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 4), plt.imshow(binary_image, cmap='gray'), plt.title("Binary Image")
    plt.xticks([]), plt.yticks([])

    plt.show()



















if __name__ == "__main__":

    main()