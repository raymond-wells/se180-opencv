# -*- coding: utf-8 -*-
import cv2


class ORBAlgorithm:

    def vectorize(self, image_file):
        orb = cv2.ORB()
        image = cv2.imread(image_file, 0)
        kp, desc = orb.detectAndCompute(image, None)
        # Now unroll the desc array...
        retval = []
        for y in desc:
            for x in y:
                retval.append(int(x))
        return retval

    def preprocess(self, image_file):
        orb = cv2.ORB()
        image = cv2.imread(image_file, 0)
        kp, desc = orb.detectAndCompute(image, None)
        return desc
