import cv2
import argparse
import os
import numpy as np
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required = True, help = "Path to the template file.")
ap.add_argument("-o", "--output", required = True, help = "Path to the output mask folder.")
ap.add_argument("-c", "--color", required = True, help = "BGR color to filter the field's masks.")
args = vars(ap.parse_args())

# HSV field color
color = args['color'].split(',')

if len(color) != 3:
    raise ValueError('HSV color error.', args['color'])

color = np.array(color, dtype='uint8')

# Create output folder
os.makedirs(args['output'], exist_ok=True)

# Extract template file name
(_, template_name) = os.path.split(args['template'])

# Filter from the template only the masks
template = cv2.imread(args['template'])

# Filter from the template only the masks
fields_mask = cv2.inRange(template, (color), (color))
print(fields_mask.shape)

# Grab the contour of the fields
cnts = cv2.findContours(fields_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Show the contours
clone = template.copy()
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)

# Template mask
mask = np.zeros(template.shape[:2], dtype='uint8')

# Loop over each field's mask
for c in cnts:

    # Fit a rotated bounding box to the contour and draw a rotated bounding box
    box = cv2.minAreaRect(c)
    box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))

    # Copy the mask
    cv2.drawContours(mask, [box], -1, 255, -1)

# Save the mask of the template
output_file = os.path.join(args['output'], template_name)
cv2.imwrite(output_file, mask)

print('Finished.')