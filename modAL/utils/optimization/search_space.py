import numpy as np


class SearchSpace:
    __slots__ = ('space', 'var_names')

    def __init__(self, *args, **kwargs):
        self.space = list(args) + list(kwargs.values())
        self.var_names = list(kwargs.keys())

    def __getitem__(self, idx):
        return self.space[idx]()

    def numpy_sample(self):
        return np.array([dist() for dist in self.space])
