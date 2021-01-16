# First import OpenCV, NumPY and MatPlotLib as we will use these libraries
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import * 


# Let's use starfish.  I've picked up a blurred version of our starfish
# to make it a little easier for K-means to segment into two clusters.

img = Image.open("Images/starfish_blur.png")

# Store the image's width and height for later use. 
imgWidth = img.size[0]
imgHeight = img.size[1]

# We're going to use 7 features to segment
# This is an experimental choice.
# Choosing and normalising features can be a matter of experimentation.

numFeatures = 7
# Create a data vector, with 7 values
#  blue
#  green
#  red
#  x
#  y
#  red - blue and
#  red - green 
#
# for every pixel in the image

# Initially I used 5 features, but when I added red-blue and red-green
# the clustering improved.

Z = np.ndarray(shape=(imgWidth * imgHeight, numFeatures), dtype = float)

# You must use float32 here as 'float' does not 
# have to be 32-bit. float32 has to be 32-bit
# and we'll shortly use a routine that needs float32s.
Z = np.float32(Z)

K = 2

# Create our cluster centers.

# clusterCenter is an arrangement of 'K' vectors
# each vector 'contains' our 7 columns or features that we described 
# in the preceding code block.
# For example, eventually, clusterCenters[0][0] will contain
# the mean of the 'blue's in cluster 0 and clusterCenters[0][1] will
# contain the mean of the 'red's in cluster 0, etc.
clusterCenters = np.ndarray(shape=(K,numFeatures))

# Initialise each element of both of our vectors 
# to rand values (each random number being between the max'es & mins of that feature in Z)
maxVals = np.amax(Z)
minVals = np.amin(Z)
for i, _ in enumerate(clusterCenters):
        clusterCenters[i] = np.random.uniform(minVals, maxVals, numFeatures)

# Load data vector with the 7 values
for y in tqdm(range(0, imgHeight), ascii=True):
    for x in range(0, imgWidth):
        xy = (x, y)
        rgb = img.getpixel(xy)
        Z[x + y * imgWidth, 0] = rgb[0]           # blue
        Z[x + y * imgWidth, 1] = rgb[1]           # green
        Z[x + y * imgWidth, 2] = rgb[2]           # red
        # Experimentally, reduce the influence of the x,y components by dividing them by 10
        Z[x + y * imgWidth, 3] = x / 10           # x
        Z[x + y * imgWidth, 4] = y / 10           # y 
        Z[x + y * imgWidth, 5] = rgb[2] - rgb[0]  # red - blue
        Z[x + y * imgWidth, 6] = rgb[2] - rgb[1]  # red - green

# We need a second copy of our initial vector
# for OpenCV's K-means implementation.
Z2 = Z.copy()

# OpenCV's K-means 
criteria = (cv2.TERM_CRITERIA_MAX_ITER, i+1, 0.1)
ret, label, center = cv2.kmeans(Z2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Convert center back into unsigned bytes
center = np.uint8(center)

# reshape the RGB values from our cv2.kmeans results into
# an image.
rgb = center[:,0:3]
res = rgb[label.flatten()]
img = res.reshape((imgHeight,imgWidth, 3))

plt.imshow(img)
