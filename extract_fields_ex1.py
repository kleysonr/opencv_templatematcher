import cv2
import os
import argparse
import glob
import numpy as np
import imutils
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from libs.features import DetectAndDescribe, ImageMatcher

"""
This example will not change the image base to avoid any kind of distortion after
a perspective change. 

Instead, we will calculate the homography matrix between the template and the image
and apply a perspective transformation on the mask.

Then we will be able to apply the transformed mask against the original image to 
extract the all the needed information.
"""

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required = True, help = "Path to the template file.")
ap.add_argument("-m", "--mask", required = True, help = "Path to the output mask folder.")
ap.add_argument("-i", "--image", required = True, help = "Path to the image file.")
args = vars(ap.parse_args())

# Extract template file name
(_, template_name) = os.path.split(args['template'])

# Mask file
mask_file = os.path.join(args['mask'], template_name)

# Initialize the keypoint detector, local invariant descriptor and descriptor
detector = FeatureDetector_create('SIFT')
descriptor = DescriptorExtractor_create('RootSIFT')
dad = DetectAndDescribe(detector, descriptor)
im = ImageMatcher(dad, glob.glob('templates' + "/*.jpg"))

# Read image
image = cv2.imread(args['image'])
image = imutils.resize(image, width=800)

# Read template
template = cv2.imread(args['template'])

# Read template's mask
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
(T, mask) = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

# Due to noise in the loaded mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Transform the image source
transformed = im.transform(template, image, mask=mask, width=800)

# Grab the contours for the fields
cnts = cv2.findContours(transformed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Loop over each field to extract the information
for i, c in enumerate(cnts):

    # Fit a bounding box to the contour
    (x, y, w, h) = cv2.boundingRect(c)

    # Extract the ROI
    roi = image[y:y+h, x:x+w]

    # Show it
    cv2.imshow('ROI #{}'.format(i), roi)

cv2.imshow('Image', image)
cv2.waitKey(0)


