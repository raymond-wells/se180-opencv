from orbalgorithm import ORBAlgorithm
from v1l import V1LikeAlgorithm
from briskalgorithm import BRISKAlgorithm
class AlgorithmFactory:
    def __init__(self):
        self.algos = { "orb" : ORBAlgorithm,
                       "v1l" : V1LikeAlgorithm,
                       "brisk" : BRISKAlgorithm}
    def get(self, algo):
        return self.algos[algo];
