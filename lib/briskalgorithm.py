# -*- coding: utf-8 -*-
import cv2


class BRISKAlgorithm:
    def __init__(self):
        self.needs_bof = True
    def vectorize(self, image_file):
        orb = cv2.BRISK()
        image = cv2.imread(image_file, 0)
        kp, desc = orb.detectAndCompute(image, None)
        # Now unroll the desc array...
        retval = []
        for y in desc:
            for x in y:
                retval.append(int(x))
        return retval

    def preprocess(self, image_file):
        orb = cv2.BRISK()
        image = cv2.imread(image_file, 0)
        kp, desc = orb.detectAndCompute(image, None)
        return desc
