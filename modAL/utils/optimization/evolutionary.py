import numpy as np
from copy import deepcopy
from itertools import product
from collections import Container
from multiprocessing import Pool, cpu_count


class Specimen:

    def __init__(self, search_space):
        self.search_space = search_space
        self.genes = self.search_space.sample()

    def __repr__(self):
        return "Specimen(%s)" % str(self.genes)

    def __getitem__(self, idx):
        return self.genes[idx]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __len__(self):
        return len(self.genes)

    def __eq__(self, other):
        assert isinstance(other, Specimen), 'the objects to compare must be an instance of Specimen'

        return np.all(self.genes == other.genes)

    def mutate(self, max_mutations=5):
        pass

    def make_offspring(self, father):
        pass
