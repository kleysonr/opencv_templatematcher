import cv2
import glob
from libs.features import DetectAndDescribe, ImageMatcher
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create

# Initialize the keypoint detector, local invariant descriptor and descriptor
detector = FeatureDetector_create('SIFT')
descriptor = DescriptorExtractor_create('RootSIFT')
dad = DetectAndDescribe(detector, descriptor)
im = ImageMatcher(dad, glob.glob('masks' + "/*.jpg"))

# Loop over all the sample images
for f in glob.glob('images' + "/*.jpg"):

    queryImage = cv2.imread(f)
    grayImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # Matching score for each mask file
    masks = im.search(grayImage)
    print(masks)

    if len(masks) != 0:

        # Show the matched keypoints
        mask = cv2.imread(masks[0][1])
        visImage = im.show(queryImage, mask, width=800)

        # Transform image source
        transformedM = im.transform(mask, queryImage, width=800)
        transformedI = im.transform(queryImage, mask, width=800)

        cv2.imshow('Show', visImage)
        cv2.imshow('Transformed Mask', transformedM)
        cv2.imshow('Transformed Image', transformedI)
        cv2.waitKey(0)
