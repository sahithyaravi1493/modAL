import numpy as np
from copy import deepcopy
from itertools import product
from collections import Container
from multiprocessing import Pool, cpu_count


class Specimen:
    __slots__ = ('search_space', 'genes')

    def __init__(self, search_space, initial_genes=None):
        self.search_space = search_space
        if initial_genes is None:
            self.genes = self.search_space.numpy_sample()
        elif isinstance(initial_genes, np.ndarray):
            self.genes = initial_genes

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
        n_mutations = np.random.randint(1, min(max_mutations+1, len(self.genes)+1))
        mutation_idx = np.random.choice(range(len(self.genes)), size=n_mutations, replace=False)
        self.genes[mutation_idx] = self.search_space[mutation_idx]

    def make_offspring(self, father, p_keep=0.5):
        assert 0 <= p_keep <= 1, 'p_keep must be between one and zero'
        assert len(father) == len(self.genes), 'the two Specimens to be crossed must contain the same number of genes'

        mother_gene_mask = np.random.choice([True, False], p=[p_keep, 1-p_keep], size=len(self.genes))
        offspring_genes = self.genes*mother_gene_mask + father.genes*(~mother_gene_mask)

        return Specimen(self.search_space, offspring_genes)

