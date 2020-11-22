
import matplotlib.pyplot as plt
import pydicom

dataset = pydicom.dcmread("DICOM_images/1-039.dcm")
img = dataset.pixel_array
plt.imshow(img, cmap="gray")
plt.imsave("converted.jpg",img,cmap="gray")
