"""
Optimizer interface for query synthesis and acquisition function optimization.
"""

from functools import partial


def arg2Dto1D(func):
    """
    Decorator to convert numpy array arguments between scikit-learn and
    scipy.minimize conventions. When optimizing a function calling scikit-learn
    estimator methods, arguments needs to have 2D shape. On the other hand, when
    optimizing functions with scipy.minimize, the argument needs to be 1D. This
    decorator converts between the two conventions.

    :param func: Function for which the numpy array argument needs to be converted.
    :type func: callable
    :returns:
      - **input_converted_func** *(callable)* --
        Function which handles numpy array arguments according to scikit-learn
        conventions.
    """

    def input_converted_func(X, *args, **kwargs):
        result = func(X.reshape(1, -1), *args, **kwargs)
        return result.reshape(1, -1)

    return input_converted_func


class UtilityOptimizer:
    def __init__(self, utility_measure, optimizer_func, *opt_args, **opt_kwargs):
        self.utility_measure = utility_measure
        self.optimizer_func = optimizer_func

    def __call__(self, model, *optimizer_args, **optimizer_kwargsx):
        model_utility = partial(self.utility_measure, model)
        result = self.optimizer_func(model_utility, *optimizer_args, **optimizer_kwargsx)
        return result
