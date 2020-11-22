import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import feature
from scipy.ndimage import distance_transform_edt
from skimage import measure, color
from skimage import morphology, segmentation
from sklearn.cluster import KMeans
import cv2
from skimage.segmentation import clear_border

#coins = data.coins()
coins = cv2.imread("1-039.jpg",0)
plt.figure(1)
plt.imshow(coins,cmap="gray")

coins_denoised = filters.median(coins, selem=np.ones((5,5)))
f, (ax0,ax1) = plt.subplots(1,2,figsize=(15,5))
ax0.imshow(coins)
ax1.imshow(coins_denoised)

edges = skimage.feature.canny(coins_denoised,sigma=4)#sigma adjusts the width of canny, how sensitive it is
#sigma ne kadar küçükse o kadar etkili ve hassas, ayrıntılı, more noisy
#(thresh, edges) = cv2.threshold(coins_denoised, 100, 255, cv2.THRESH_BINARY)


print(type(edges),edges.shape,edges)

#kernel = np.ones((3, 3), np.uint8)
#img_erosion = cv2.erode(edges, kernel, iterations=1)
#edges = cv2.dilate(img_erosion, kernel, iterations=1)
#edges = img_dilation
#edges = skimage.feature.canny(edges,sigma=3)
#edges = clear_border(opening)
plt.figure(3)
plt.imshow(edges,cmap="gray")
edges_not = np.logical_not(edges)
dt = distance_transform_edt(edges_not)
plt.figure(4)
plt.imshow(dt)

local_max = feature.peak_local_max(dt,indices=False,min_distance=5)
plt.figure(5)
plt.imshow(local_max,cmap="gray")
print(local_max)
peak_idx = feature.peak_local_max(dt,indices=True,min_distance=5)
plt.figure(7)
plt.plot(peak_idx[:,1],peak_idx[:,0],"r.")
plt.imshow(dt)

markers = measure.label(local_max)
print(markers)
labels = morphology.watershed(-dt,markers)
plt.figure(8)
plt.imshow(color.label2rgb(labels,image=coins_denoised))
plt.figure(9)
plt.imshow(color.label2rgb(labels,image=coins_denoised,kind="avg"),cmap="gray")
regions = measure.regionprops(labels,intensity_image=coins_denoised)
region_means = [r.mean_intensity for r in regions]

plt.figure(10)
plt.hist(region_means,bins=20)

model = KMeans(n_clusters=2)
region_means = np.array(region_means).reshape(-1,1)
model.fit(np.array(region_means).reshape(-1,1))
print(model.cluster_centers_)
bg_fg_labels = model.predict(region_means)
classified_labels = labels.copy()
print(classified_labels)
print(bg_fg_labels)
for bg_fg, region in zip(bg_fg_labels,regions):
    classified_labels[tuple(region.coords.T)] = bg_fg
plt.figure(11)
plt.imshow(color.label2rgb(classified_labels,image=coins_denoised))
peak_idx = feature.peak_local_max(classified_labels,indices=True,min_distance=15)
print(peak_idx)





plt.show()