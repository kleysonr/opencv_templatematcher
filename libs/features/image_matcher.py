import cv2
import numpy as np
import imutils
from imutils.feature import DescriptorMatcher_create

class ImageMatcher:

    def __init__(self, detectAndDescribe, masksPaths, minMatches=1):
        """[summary]
        
        Arguments:
            detectAndDescribe {[type]} -- [description]
            masksPaths {[type]} -- [description]
        
        Keyword Arguments:
            minMatches {int} -- The minimum number of matches required for a homography to be calculated. (default: {1})
        """

        self.dad = detectAndDescribe
        self.masksPaths = masksPaths
        self.minMatches = minMatches

    def search(self, queryImage):
        """
        This method will take the keypoints and descriptors from the query image
        and then match them against a database of keypoints and descriptors extracted
        from the mask files. The entry in the database with the best match will be
        chosen as the identification for the mask to be applied.
        
        Arguments:
            queryImage {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        # Convert to grayscale
        queryImage = self.getGray(queryImage)

        # Describe the query image
        (queryKps, queryDescs) = self.dad.describe(queryImage)

        # Initialize the dictionary of results
        results = {}

        # Loop over all the mask images
        for maskPath in self.masksPaths:

            # Load the mask image
            mask = cv2.imread(maskPath)

            # Convert to grayscale
            mask = self.getGray(mask)

            # Extract its kps and decriptors
            (kps, descs) = self.dad.describe(mask)

            # Get a score of matched keypoints and update the results
            matches = self.match(queryDescs, descs)
            score = self.score(queryKps, kps, matches)
            results[maskPath] = score

        # Sort the matches in DESC order
        if len(results) > 0:
            results = sorted([(v, k) for (k, v) in results.items() if v > 0], reverse = True)

        return results

    def match(self, featuresA, featuresB, ratio=0.7):
        """[summary]
        
        Arguments:
            featuresA {[type]} -- The list of feature vectors associated with the first image to be matched.
            featuresB {[type]} -- The list of feature vectors associated with the second image to be matched.
        
        Keyword Arguments:
            ratio {float} -- The ratio of nearest neighbor distances suggested by David Lowe (creator of the SIFT algorithm) to prune down the number of keypoints a homography needs to be computed for. (default: {0.7})
        
        Returns:
            [type] -- [description]
        """

        # Compute the raw matches and initialize the list of actual matches
        matcher = DescriptorMatcher_create('BruteForce')
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:

            # Ensure the distance is within a certain ratio of each other
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        return matches

    def score(self, kpsA, kpsB, matches):
        """[summary]
        
        Arguments:
            kpsA {[type]} -- The list of keypoints associated with the first image to be matched.
            kpsB {[type]} -- The list of keypoints associated with the second image to be matched.
            matches {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        # Check to see if there are enough matches to process
        if len(matches) > self.minMatches:

            # Construct the two sets of points
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])
 
            # Compute the homography between the two sets of points
            # and compute the ratio of matched points
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
 
            # Return the ratio of the number of matched keypoints
            # to the total number of keypoints
            return float(status.sum()) / status.size

        # No matches were found
        return -1

    def show(self, queryImage, maskImage, width=None):
        """[summary]
        
        Arguments:
            queryImage {[type]} -- [description]
            maskImage {[type]} -- [description]
        
        Keyword Arguments:
            width {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """

        # Resize images for the same size
        (queryImage, maskImage) = self.doImagesSameSize(queryImage, maskImage, width=width)

        # Convert images to grayscale
        grayImage = self.getGray(queryImage)
        grayMask = self.getGray(maskImage)

        # Extract its kps and decriptors from the query image
        (queryKps, queryDescs) = self.dad.describe(grayImage)

        # Extract its kps and decriptors from the mask image
        (kps, descs) = self.dad.describe(grayMask)

        # Get keypoints that match
        matches = self.match(queryDescs, descs)

        # Create a ph to join the images
        (imageH, imageW) = maskImage.shape[:2]
        vis = np.zeros((imageH, 2 * imageW, 3), dtype="uint8")

        # Do a kind of np.hstack()
        vis[0:imageH, 0:imageW] = queryImage
        vis[0:imageH, imageW:] = maskImage

        # loop over the matches
        for (queryIdx, maskIdx) in matches:
                
            # generate a random color and draw the match
            color = np.random.randint(0, high=255, size=(3,))
            color = tuple(map(int, color))
            ptA = (int(queryKps[queryIdx][0]), int(queryKps[queryIdx][1]))
            ptB = (int(kps[maskIdx][0] + imageW), int(kps[maskIdx][1]))
            cv2.line(vis, ptA, ptB, color, 2)

        return vis

    def transform(self, imageA, imageB, width=None):
        """[summary]
        
        Arguments:
            imageA {[type]} -- [description]
            imageB {[type]} -- [description]
        
        Keyword Arguments:
            width {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """

        # Resize images for the same size
        (imageA, imageB) = self.doImagesSameSize(imageA, imageB, width=width)

        # Convert images to grayscale
        grayA = self.getGray(imageA)
        grayB = self.getGray(imageB)

        # Extract its kps and decriptors from the base image
        (kpsA, descsA) = self.dad.describe(grayA)

        # Extract its kps and decriptors from the image to be transformed
        (kpsB, descsB) = self.dad.describe(grayB)

        # Get keypoints that match
        matches = self.match(descsA, descsB)

        # Check to see if there are enough matches to process
        if len(matches) > self.minMatches:

            # Construct the two sets of points
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])
 
            # Compute the homography between the two sets of points
            # and compute the ratio of matched points
            (h, _) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

            # Apply the H matrix to the src image
            new_image = cv2.warpPerspective(imageA, h, (imageB.shape[1], imageB.shape[0]))

            return new_image

        return None

    def doImagesSameSize(self, imageA, imageB, width=None):
        """[summary]
        
        Arguments:
            imageA {[type]} -- [description]
            imageB {[type]} -- [description]
        
        Keyword Arguments:
            width {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """

        # Resize images for the same size
        if width is None:
            width = imageA.shape[1]

        imageB = imutils.resize(imageB, width=width)
        imageA = imutils.resize(imageA, width=width)

        if imageA.shape[0] > imageB.shape[0]:
            _img = np.zeros(imageA.shape, dtype='uint8')
            _img[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
            imageB = _img

        else:
            _img = np.zeros(imageB.shape, dtype='uint8')
            _img[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
            imageA = _img

        return (imageA, imageB)

    def getGray(self, image):
        """[summary]
        
        Arguments:
            image {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        # Convert images to grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image