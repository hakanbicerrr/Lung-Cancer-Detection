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
    #segmented_image_rgb = random_walker(image_rgb)
    image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)
    segmented_image = random_walker(image)
    filtered_image = gaussian_of_median(segmented_image)
    (thresh, binary_image) = binary_threshold(filtered_image)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imshow("binary",binary_image)

    opening = clear_border(opening)

    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    ret2, sure_fg = cv2.threshold(dist_transform, 0.03 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    # cv2.imshow("unknown",unknown)
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_rgb, markers)
    image_rgb[markers == -1] = [255, 0, 255]
    img3 = color.label2rgb(markers, bg_label=0)
    cv2.imshow("rgb",image_rgb)
    cv2.imshow("im3",img3)
    cv2.waitKey()


def binary_threshold(image):
    return cv2.threshold(image, 165, 255, cv2.THRESH_BINARY)


def gaussian_of_median(segmented_image):
    kernel = np.ones((7, 7), np.float32) / 49
    median_2d = cv2.filter2D(segmented_image, -1, kernel)
    gaussian_of_median_2d = cv2.GaussianBlur(median_2d, (5, 5), 0)
    return gaussian_of_median_2d


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

def random_walker(image):

    new_image = np.zeros((512, 512), np.uint8)
    image_labels = np.zeros(image.shape, dtype=np.uint8)
    points = circle_points(110, [255, 255], 180)[:-1]
    indices = draw.circle_perimeter(255, 255, 50)
    image_labels[indices] = 1
    image_labels[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2
    image_show(image_labels)
    image_segmented = seg.random_walker(image, image_labels, beta=2500)

    fig, ax = image_show(image)
    ax.imshow(image_segmented == 1, alpha=0.3)

    #plt.show()
    new_image[image_segmented==1] = image[image_segmented==1]
    #cv2.imshow("seg",new_image)
    #cv2.waitKey()
    return new_image



if __name__ == "__main__":

    main()