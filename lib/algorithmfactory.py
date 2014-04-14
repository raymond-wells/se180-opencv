from orbalgorithm import ORBAlgorithm
from v1l import V1LikeAlgorithm
class AlgorithmFactory:
    def __init__(self):
        self.algos = { "orb" : ORBAlgorithm,
                       "v1l" : V1LikeAlgorithm}
    def get(self, algo):
        return self.algos[algo];
