import cv2
import numpy as np
from skimage.color import label2rgb
from skimage import morphology
from skimage import feature
import scipy.ndimage as ndi
from skimage import segmentation


def imcomplement(image):
  """Equivalent to matlabs imcomplement function"""

  min_type_val = np.iinfo(np.uint8).min
  max_type_val = np.iinfo(np.uint8).max
  return min_type_val + max_type_val - image

def imregionalmax(image, ksize=3):
  """Similar to matlab's imregionalmax"""
  filterkernel = np.ones((ksize, ksize)) # 8-connectivity
  reg_max_loc = feature.peak_local_max(image,
                               footprint=filterkernel, indices=False,
                               exclude_border=0)
  return reg_max_loc.astype(np.uint8)

image = cv2.imread("1-039.jpg",0)
sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)

grad_magnitude = np.sqrt(sobelx**2 + sobely**2)
grad_magnitude = 255*(grad_magnitude-np.min(grad_magnitude)) / (np.max(grad_magnitude) - np.min(grad_magnitude))

kernel = np.ones((3, 3), np.uint8)

opened = cv2.dilate(image, kernel, iterations=2)
opened = cv2.erode(opened, kernel, iterations=2)

selem = morphology.disk(1)
#opened = morphology.opening(image, selem)
eroded = cv2.erode(image, kernel, iterations=2)
cv2.imshow("opened",opened)
#eroded = morphology.erosion(image, selem)
cv2.imshow("eroded",eroded)
opening_recon = morphology.reconstruction(seed=eroded, mask=image, method='dilation')
closed_opening = morphology.closing(opened, selem)
dilated_recon_dilation = morphology.dilation(opening_recon, selem)
recon_erosion_recon_dilation = morphology.reconstruction(dilated_recon_dilation,
                                                    opening_recon,
                                                    method='erosion').astype(np.uint8)

recon_dilation_recon_dilation = morphology.reconstruction(imcomplement(dilated_recon_dilation),
                                                     imcomplement(opening_recon),
                                                     method='dilation').astype(np.uint8)
recon_dilation_recon_dilation_c = imcomplement(recon_dilation_recon_dilation)

print(np.linalg.norm(recon_erosion_recon_dilation - recon_dilation_recon_dilation_c))
# Outputs 0.0
print(np.allclose(recon_erosion_recon_dilation, recon_dilation_recon_dilation_c))
# Outputs True
foreground_1 = imregionalmax(recon_erosion_recon_dilation, ksize=65)
fg_superimposed_1 = image.copy()
fg_superimposed_1[foreground_1 == 1] = 255
foreground_2 = morphology.closing(foreground_1, np.ones((5, 5)))
foreground_3 = morphology.erosion(foreground_2, np.ones((5, 5)))
foreground_4 = morphology.remove_small_objects(foreground_3.astype(bool), min_size=20)

_, labeled_fg = cv2.connectedComponents(foreground_4.astype(np.uint8))
col_labeled_fg = label2rgb(labeled_fg)
_, thresholded = cv2.threshold(recon_erosion_recon_dilation, 0, 255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
labels = morphology.watershed(grad_magnitude, labeled_fg)
superimposed = image.copy()
watershed_boundaries = segmentation.find_boundaries(labels)
superimposed[watershed_boundaries] = 255
superimposed[foreground_4] = 255
cv2.imshow("super",superimposed)
cv2.waitKey()