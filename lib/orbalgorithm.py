# -*- coding: utf-8 -*-
import cv2


class ORBAlgorithm:

    def vectorize(self, image):
        orb = cv2.ORB()
        kp = orb.detect(image, None)
        kp, desc = orb.compute(image, kp)
        # Now unroll the desc array...
        retval = []
        for y in desc:
            for x in y:
                retval.append(int(x))
        return retval