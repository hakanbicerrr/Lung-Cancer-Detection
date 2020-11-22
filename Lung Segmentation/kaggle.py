import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io,color,measure
from skimage.segmentation import clear_border
from skimage import segmentation




def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed

image_rgb = cv2.imread("1-039 (2).jpg")
image = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)
# Show some example markers from the middle
test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(image)
print("Internal Marker")
plt.imshow(test_patient_internal, cmap='gray')
plt.show()
print("External Marker")
plt.imshow(test_patient_external, cmap='gray')
plt.show()
print("Watershed Marker")
plt.imshow(test_patient_watershed, cmap='gray')
plt.show()