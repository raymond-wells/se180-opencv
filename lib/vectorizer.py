import cv2


class Vectorizer:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def vectorize(self, image_file):
        image = cv2.imread(image_file, 0)
        return self.algorithm.vectorize(image)

    def preprocess(self, image_file):
        image = cv2.imread(image_file, 0)
        return self.algorithm.preprocess(image)