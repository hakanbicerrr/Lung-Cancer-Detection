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
def main():

    image_rgb = cv2.imread("1-039.jpg")
    image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)

    #image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
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

    img_erosion = cv2.erode(binary_image, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    print(contours)
    print(len(contours))
    '''for count in contours:
        rect = cv2.minAreaRect(count)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(image,[box],0,(0,255,0),3)
    '''
    '''for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        img = cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
    '''
    cv2.drawContours(image,contours,-1,(0,255,0),2)
    cv2.imshow("sa",image)
    edges = feature.canny(img_dilation, sigma=1)
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 3),
                                        sharex=True, sharey=True)
    ax1.imshow(edges,cmap="gray")
    plt.show()
    edges = np.logical_not(edges)
    dt = distance_transform_edt(edges)
    cv2.imshow("dt",dt)
    cv2.waitKey()



    local_max = feature.peak_local_max(dt, indices=False, min_distance=2)
    peak_idx = feature.peak_local_max(dt, indices=True, min_distance=2)
    markers = measure.label(local_max)
    print(markers)
    labels = morphology.watershed(-dt, markers)
    regions = measure.regionprops(labels, intensity_image=img_dilation)
    region_means = [r.mean_intensity for r in regions]
    model = KMeans(n_clusters=2)
    region_means = np.array(region_means).reshape(-1, 1)
    model.fit(np.array(region_means).reshape(-1, 1))
    print(model.cluster_centers_)
    bg_fg_labels = model.predict(region_means)
    classified_labels = labels.copy()
    print(classified_labels)
    print(bg_fg_labels)
    for bg_fg, region in zip(bg_fg_labels, regions):
        classified_labels[tuple(region.coords.T)] = bg_fg
    plt.figure(11)
    plt.imshow(color.label2rgb(classified_labels, image=img_dilation))







    #img_dilation = cv2.GaussianBlur(img_dilation, (3, 3), 0)
    #mask = img_dilation == 255
    #s = [[1,1,1],[1,1,1],[1,1,1]]
    #label_mask, num_labels = ndimage.label(mask,structure=s)
    #print(label_mask.shape,num_labels)
    #img2 = color.label2rgb(label_mask,bg_label=0)
    #cv2.imshow("colored labels",img2)
    #clusters = measure.regionprops(label_mask,image)
    #print(len(clusters))
    #for prop in clusters:
    #    print("Label: {} Area: {} Perim: {}".format(prop.label,prop.area,prop.perimeter))
    #ret, otsu = cv2.threshold(gaussian_of_median_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #img_canny = cv2.Canny(otsu, 100, 200)
    #img_sobelx = cv2.Sobel(otsu, cv2.CV_8U, 1, 0, ksize=3)
    #img_sobely = cv2.Sobel(otsu, cv2.CV_8U, 0, 1, ksize=3)
    #img_sobel = img_sobelx + img_sobely
    #img_sobel = sobel_operator(image)
    '''kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_rgb, markers)
    image_rgb[markers == -1] = [255, 0, 0]
'''
    '''plt.subplot(3,3,1), plt.imshow(image,cmap='gray'), plt.title("Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 3), plt.imshow(median_2d_5, cmap='gray'), plt.title("Median Filtered Image 5x5")
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 5), plt.imshow(gaussian_of_median_2d_5, cmap='gray'), plt.title("Gaussian of Median Image 5x5")
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 6), plt.imshow(binary_image, cmap='gray'), plt.title("Binary Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 7), plt.imshow(img_erosion, cmap='gray'), plt.title("Binary Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 8), plt.imshow(img_dilation, cmap='gray'), plt.title("Binary Image")
    plt.xticks([]), plt.yticks([])
    '''


    #plt.subplot(3, 3, 7), plt.imshow(otsu, cmap='gray'), plt.title("Binary Image")
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(2,3,2), plt.imshow(average_image,cmap='gray'), plt.title("Average Filtered Image")
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(2,3,2), plt.imshow(gaussian_image, cmap='gray'), plt.title("Gaussian Filtered Image")
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(2,3,3), plt.imshow(median_image, cmap='gray'), plt.title("Median Filtered Image")
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(2,3,4), plt.imshow(gaussian_of_median_image, cmap='gray'), plt.title("Gaussian of Median Image")
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(4,4,6), plt.imshow(otsu, cmap='gray'), plt.title("Binarized Image")
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(4,4,7), plt.imshow(img_canny, cmap='gray'), plt.title("Binarized Image")
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(4,4,8), plt.imshow(img_sobel, cmap='gray'), plt.title("Binarized Image")
    #plt.xticks([]), plt.yticks([])

    '''plt.subplot(4,4,9), plt.imshow(opening, cmap='gray'), plt.title("Binarized Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(4,4,10), plt.imshow(sure_bg, cmap='gray'), plt.title("Binarized Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(4,4,11), plt.imshow(dist_transform, cmap='gray'), plt.title("Binarized Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(4,4,12), plt.imshow(sure_fg, cmap='gray'), plt.title("Binarized Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 4, 13), plt.imshow(unknown, cmap='gray'), plt.title("Binarized Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 4, 14), plt.imshow(markers, cmap='gray'), plt.title("Binarized Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(4, 4, 15), plt.imshow(image_rgb, cmap='gray'), plt.title("Binarized Image")
    plt.xticks([]), plt.yticks([])'''
    plt.show()




def sobel_operator(image):
    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    rows, cols = image.shape
    print("Original Image Dimensions: ", rows, cols)
    new_image = np.zeros((rows,cols),  np.uint8)
    convolved_x = np.zeros((rows, cols))
    convolved_y = np.zeros((rows, cols))
    sa = 1
    size = 3
    gx = np.array([[-1,0,1],#x direction kernel
          [-2,0,2],
          [-1,0,1]])

    gy = np.array([[-1,-2,-1],#y direction kernel
          [0, 0, 0],
          [1, 2, 1]])
    # Kernel Convolution Algorithm
    for i in range(sa, rows - sa):
        for j in range(sa, cols - sa):
            for k in range(size):
                for l in range(size):
                    convolved_x[i][j] += image[i + k - sa][j + l - sa] * gx[k][l]
                    convolved_y[i][j] += image[i + k - sa][j + l - sa] * gy[k][l]
    #Sobel Operator
    convolved_x = np.square(convolved_x)
    convolved_y = np.square(convolved_y)
    gradient_magnitude = np.sqrt(convolved_x+convolved_y)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()#scale values to 0-255
    new_image = np.uint8(gradient_magnitude)

    print(np.amin(new_image),np.amax(new_image))
    return new_image



















if __name__ == "__main__":

    main()