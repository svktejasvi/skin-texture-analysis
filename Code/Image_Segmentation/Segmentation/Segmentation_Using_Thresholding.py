# First import OpenCV, NumPY and MatPlotLib as we will use these libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


### Begin Visualization ####
fig = plt.figure()
i = 0
def show(image,color,num_sub_plots, title, xinch, yinch, dpi):
    global i
    i += 1
    plt.subplot(num_sub_plots,num_sub_plots,i)
    fig.set_size_inches(xinch,yinch)
    fig.set_dpi(dpi)
    plt.axis('off')
    plt.title(title)
    if color == 'BGR':
        plt.imshow(image)
    elif color == 'RGB':
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif color == 'GRAY':
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    #fig.savefig(abc.png, dpi=100)

num_sub_plots = 4
xinch = 15
yinch = 10
dpi = 80
#### End Visualization ####

# Load a color image
img = cv2.imread("Images/starfish.png")
show(img, 'RGB', num_sub_plots, 'Original_Image_RGB', xinch, yinch, dpi)
show(img, 'BGR', num_sub_plots, 'Original_Image_OpenCV', xinch, yinch, dpi)

# Apply some blurring to reduce noise

# h is the Parameter regulating filter strength for luminance component. 
# Bigger h value perfectly removes noise but also removes image details, 
# smaller h value preserves details but also preserves some noise

# Hint: I recommend using larger h and hColor values than typical to remove noise at the
# expense of losing image details

# Experiment with setting h and hColor to a suitable value.

# Exercise: Insert code here to set values for h and hColor. 
h = hColor = 20
# Hint: You'll find answers at the bottom of the lab. 
    
# Default values
templateWindowSize = 7
searchWindowSize = 21   
blur = cv2.fastNlMeansDenoisingColored(img, None,h,hColor,templateWindowSize,searchWindowSize)  
show(blur, 'RGB', num_sub_plots, 'Blurred_Denoised_RGB', xinch, yinch, dpi)

# Apply a morphological gradient (dilate the image, erode the image, and take the difference

elKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))

# YOUR CODE HERE
# Exercise: Use openCV's morphologyEx to generate a gradient using the kernel above
# Hint: You'll find answers at the bottom of the lab. 
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, elKernel)
# END YOUR CODE HERE

show(gradient, 'RGB', num_sub_plots, 'Gradient_RGB_EdgeDetection', xinch, yinch, dpi)


# Apply Otsu's method - or you can adjust the level at which thresholding occurs
# and see what the effect of this is

# Convert gradient to grayscale
gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
show(gradient, 'GRAY', num_sub_plots, 'Gradient_Edge_Gray', xinch, yinch, dpi)

# YOUR CODE HERE
# Exercise: Generate a matrix called otsu using OpenCV's threshold() function.  Use
# Otsu's method.
# Hint: You'll find answers at the bottom of the lab. 
otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
show(otsu, 'GRAY', num_sub_plots, 'After_Otsu', xinch, yinch, dpi)
# END YOUR CODE HERE

       
# Apply a closing operation - we're using a large kernel here. By all means adjust the size of this kernel
# and observe the effects
closingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33,33))
close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, closingKernel)
show(close, 'GRAY', num_sub_plots, 'After_Closing', xinch, yinch, dpi)


# Erode smaller artefacts out of the image - play with iterations to see how it works
    
# YOUR CODE HERE
# Exercise: Generate a matrix called eroded using cv2.erode() function over the 'close' matrix.
# Experiment until your output image is similar to the image below
# Hint: You'll find answers at the bottom of the lab. 
eroded = cv2.erode(close, None, iterations=6)
# END YOUR CODE HERE

show(eroded, 'GRAY', num_sub_plots, 'After_Erosion', xinch, yinch, dpi)


p = int(img.shape[1] * 0.05)
eroded[:, 0:p] = 0
eroded[:, img.shape[1] - p:] = 0

show(eroded, 'GRAY', num_sub_plots, 'After_Removing_5%_Left_Right', xinch, yinch, dpi)

# YOUR CODE HERE
# Exercise: Find the contours - just external contours to keep post-processing simple
# Hint: You'll find answers at the bottom of the lab. 
(cnting, contours, _) = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# END YOUR CODE HERE


# Sort the candidates by size, and just keep the largest one
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

# Lets create two images, initially all zeros (i.e. black)
# One image will be filled with 'Blue' wherever we think there's some starfish
# The other image will be filled with 'Green' whereever we think there's not some starfish
h, w, num_c = img.shape
segmask = np.zeros((h, w, num_c), np.uint8)
stencil = np.zeros((h, w, num_c), np.uint8)

# I know we've only one contour, but - in general - we'd expect to have more contours to deal with
for c in contours:
    # Fill in the starfish shape into segmask
    cv2.drawContours(segmask, [c], 0, (255, 0, 0), -1)
    # Lets fill in the starfish shape into stencil as well
    # and then re-arrange the colors using numpy
    cv2.drawContours(stencil, [c], 0, (255, 0, 0), -1)
    stencil[np.where((stencil==[0,0,0]).all(axis=2))] = [0, 255, 0]
    stencil[np.where((stencil==[255,0,0]).all(axis=2))] = [0, 0, 0]

# Now, lets create a mask image by bitwise ORring segmask and stencil together
mask = cv2.bitwise_or(stencil, segmask)

show(mask, 'RGB', num_sub_plots, 'Masking_Largest_Contours', xinch, yinch, dpi)

# Now, lets just blend our original image with our mask

# YOUR CODE HERE
# Exercise: Blend the original image 'img' and our mask 'mask'
# in any way you see fit, and store it in a variable called output
# Hint: You'll find answers at the bottom of the lab. 
output = cv2.bitwise_or(mask, img)
# END YOUR CODE HERE

show(output, 'RGB', num_sub_plots, 'Masking_Largest_Contours', xinch, yinch, dpi)