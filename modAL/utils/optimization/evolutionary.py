import numpy as np
from copy import deepcopy
from itertools import product
from collections import Container
from multiprocessing import Pool, cpu_count


class Specimen:
    __slots__ = ('search_space', 'genes', 'iter_idx')

    def __init__(self, search_space, initial_genes=None):
        self.search_space = search_space
        self.iter_idx = 0
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

    def __iter__(self):
        self.iter_idx = -1
        return self

    def __next__(self):
        if self.iter_idx >= len(self)-1:
            raise StopIteration
        else:
            self.iter_idx += 1
            return self.search_space.var_types[self.iter_idx](self.genes[self.iter_idx])

    def reshape(self, *args, **kwargs):
        return self.genes.reshape(*args, **kwargs)

    def mutate(self, max_mutations=5):
        n_mutations = np.random.randint(1, min(max_mutations+1, len(self)+1))
        mutation_idx = np.random.choice(range(len(self)), size=n_mutations, replace=False)
        self.genes[mutation_idx] = self.search_space[mutation_idx]

    def make_offspring(self, father, p_keep=0.5):
        assert 0 <= p_keep <= 1, 'p_keep must be between one and zero'
        assert len(father) == len(self), 'the two Specimens to be crossed must contain the same number of genes'

        mother_gene_mask = np.random.choice([True, False], p=[p_keep, 1-p_keep], size=len(self))
        offspring_genes = self.genes*mother_gene_mask + father.genes*(~mother_gene_mask)

        return Specimen(self.search_space, offspring_genes)


class EvolutionaryOptimizer:

    def __init__(self, search_space, n_pool, func, parallel=True):
        assert callable(func), 'function to be optimized needs to be callable'

        self.n_pool = n_pool
        self.func = func
        self.pool_fitness = None
        self.max_fitness = -np.inf
        self.argmax = None

        if parallel:
            self.parallel_pool = Pool(processes=cpu_count())
        else:
            self.parallel_pool = None

        # assemble initial pool
        self.search_space = search_space
        self.specimen_pool = [Specimen(search_space) for _ in range(n_pool)]
        self._calculate_fitness()
        self._set_max()

    def __len__(self):
        return len(self.specimen_pool)

    def __getitem__(self, idx):
        return self.specimen_pool[idx]

    def __repr__(self):
        return str(self.specimen_pool)

    def __del__(self):
        if self.parallel_pool:
            self.parallel_pool.terminate()
            del self.parallel_pool

    def _calculate_fitness(self):
        pass

    def _set_max(self):
        pass

    def breed(self, p_mutation=0.1, max_mutations=5):
        pass

    def mutate(self, p_mutation, max_mutations=5):
        pass

    def select(self):
        pass

    def run(
            self, n_generations,
            p_mutation=0.5, max_mutations=5, n_stopping_criteria=10,
            verbose=False
    ):

        pass
