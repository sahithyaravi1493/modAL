import numpy as np
from functools import partial

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats.distributions import randint, uniform
from scipy.optimize import minimize

from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_EI
from modAL.utils.optimization.optimizer import UtilityOptimizer
from modAL.utils.optimization.search_space import SearchSpace


# load training data
iris = load_iris()
X = iris['data']
y = iris['target'].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

##################
# float parameters
##################


def float_params(length_scale: float, nu: float):
    kernel = Matern(length_scale=length_scale, nu=nu)
    gpr = GaussianProcessClassifier(kernel=kernel)
    gpr.fit(X_train, y_train)
    return gpr.score(X_test, y_test)

# obtaining some initial values
X_initial = (1.0, 1.0)
y_initial = float_params(*X_initial).reshape(1, -1)
X_initial = np.array(X_initial).reshape(1, -1)

optimizer = UtilityOptimizer(optimizer_EI, partial(minimize, method='L-BFGS-B'), convert_numpy_arg=True)
bayesopt = BayesianOptimizer(
    GaussianProcessRegressor(), optimizer,
    X_training=X_initial, y_training=y_initial
)

n_queries = 10
for idx in range(n_queries):
    query_inst = bayesopt.query(np.array([0, 0])).x
    func_val = float_params(*list(query_inst))


exit()
#bayesopt.query()

####################################
# mixed integer and float parameters
####################################


# define the function to be optimized
def mixed_params(n_estimators: int, max_features: int, min_weight_fraction_leaf: float):
    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_weight_fraction_leaf=min_weight_fraction_leaf
    )
    rfc.fit(X_train, y_train)
    return rfc.score(X_test, y_test)


# define the search space
search_space = SearchSpace(
    n_estimators=partial(randint.rvs, low=1, high=10),
    max_features=partial(randint.rvs, low=1, high=3),
    min_weight_fraction_leaf=partial(uniform.rvs, loc=0.0, scale=0.5)
)

"""
# sketch of the optimizer function
def EI_optimizer(regressor, search_space, **optimizer_params):  # fill in evolutionary optimizer parameters later
    ev_opt = EvolutionaryOptimizer(func, **init_params)
    ev_opt.run(**evolve_params)
"""