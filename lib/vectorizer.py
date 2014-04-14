import cv2


class Vectorizer:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def vectorize(self, image_file):
        return self.algorithm.vectorize(image_file)

    def preprocess(self, image_file):
        return self.algorithm.preprocess(image_file)
