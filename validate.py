import cv2
import glob
from libs.features import DetectAndDescribe, ImageMatcher
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create

# Initialize the keypoint detector, local invariant descriptor and descriptor
detector = FeatureDetector_create('SIFT')
descriptor = DescriptorExtractor_create('RootSIFT')
dad = DetectAndDescribe(detector, descriptor)
im = ImageMatcher(dad, glob.glob('templates' + "/*.jpg"))

# Loop over all the sample images
for f in glob.glob('images' + "/*.jpg"):

    queryImage = cv2.imread(f)
    grayImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    # Matching score for each mask file
    templates = im.search(grayImage)
    print(templates)

    if len(templates) != 0:

        # Show the matched keypoints
        template = cv2.imread(templates[0][1])
        visImage = im.show(queryImage, template, width=800)

        # Transform image source
        transformedT = im.transform(template, queryImage, width=800)
        transformedI = im.transform(queryImage, template, width=800)

        cv2.imshow('Show', visImage)
        cv2.imshow('Transformed template', transformedT)
        cv2.imshow('Transformed Image', transformedI)
        cv2.waitKey(0)
