import numpy as np

class DetectAndDescribe:

    def __init__(self, detector, descriptor):

        self.detector = detector
        self.descriptor = descriptor

    def describe(self, image, useKpList=True):

        # Detect keypoints in the image
        kps = self.detector.detect(image)

        # If no kps found, return None
        if len(kps) == 0:
            return (None, None)

        # Extract local invariant descriptors
        (kps, descs) = self.descriptor.compute(image, kps)

        # Convert keypoints to a Numpy array
        if useKpList:
            kps = np.int0([kp.pt for kp in kps])

        # Return a tuple of keypoints and descriptors
        return (kps, descs)