"""
## Distribution dynamics

A user following distribution dynamics with a modified hedge structure

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from cyipopt import minimize_ipopt

class UserHedge:
    def __init__(self, dim: int, lambda1: float, lambda2: float, epsilon: float, budget: float):
        """
        :param dim: dimension of the state, i.e., the number of choices
        :param lambda1: combination coefficient in the modified hedge dynamics
        :param lambda2: combination coefficient in the modified hedge dynamics
        :param epsilon: step size/coefficient in the hedge dynamics
        :param budget: total budget of the loss vector
        """
        # problem and dynamics parameters
        self.dim = dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epsilon = epsilon
        self.budget = budget

        # preference state
        # self.p_init = normalize_simplex(np.ones((dim, 1)))  # uniform initial distribution
        self.p_init = normalize_simplex(np.random.rand(dim, 1))
        self.p_cur = self.p_init

        # calculate the optimal solution via feedforward numerical optimization
        self.opt_pt, self.opt_val = self.maximize_obj_scipy()
        pass

    def per_step_dynamics(self, dec: np.ndarray) -> np.ndarray:
        """
        Implement the modified hedge dynamics at every time step
        :param: dec: the current loss vector, which serves as the input
        :return: the preference state at the next time step
        """
        # convex combination
        self.p_cur = (self.lambda1 * self.p_cur + self.lambda2 * self.softmax_vec(dec)
                      + (1 - self.lambda1 - self.lambda2) * self.p_init)
        return self.p_cur

    def softmax_vec(self, dec: np.ndarray) -> np.ndarray:
        """
        Calculate the softmax function of the given decision (i.e., loss vector)
        Formula: exp(-epsilon * dec) / sum(exp(-epsilon * dec))
        :param dec: the current loss vector, which serves as the input
        :return: vector from the softmax function
        """
        dec_shift = -dec - np.max(-dec)  # shift for numerical stability
        exp_weight = np.exp(self.epsilon * dec_shift)
        exp_weight /= np.sum(exp_weight)
        return exp_weight

    def steady_state(self, dec: np.ndarray) -> np.ndarray:
        """
        Calculate the steady-state preference state corresponding to the given decision
        :param dec:
        :return: steady-state preference state
        """
        p_ss = ((self.lambda2 * self.softmax_vec(dec) + (1 - self.lambda1 - self.lambda2) * self.p_init)
                / (1 - self.lambda1))
        return p_ss

    def naive_dec_utility(self) -> np.ndarray:
        """
        Evaluate the steady-state utility corresponding to the naive decision
        :return: steady-state utility of the naive decision
        """
        # Use a greedy decision, i.e., allocating all the budget to the most probable element
        dec_naive = self.budget * (self.p_init == np.max(self.p_init)).astype(int)
        return self.steady_state(dec_naive).T @ dec_naive

    def objective_function(self, dec: np.ndarray) -> float:
        dec = dec.reshape(-1, 1)
        steady_state = self.steady_state(dec)
        return -steady_state.T @ dec  # Negative because we are maximizing

    def constraint_sum(self, dec: np.ndarray) -> float:
        return np.sum(dec) - self.budget

    def maximize_obj_scipy(self):
        """
        Optimize the objective function (i.e., the inner product) through scipy
        :return: optimal decision and optimal value
        """
        # Initial guess
        x0 = np.ones(self.dim) * self.budget / self.dim

        # Constraints
        constraints = {'type': 'eq', 'fun': self.constraint_sum}

        # Bounds for decision variables
        bounds = [(0, None) for _ in range(self.dim)]

        # Optimize
        result = minimize(self.objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimal_dec = result.x
            optimal_value = -result.fun  # Negate to get the original objective value
            print(f'The optimal decision is {optimal_dec} with the optimal value {optimal_value}.')
            return optimal_dec.reshape(-1, 1), optimal_value
        else:
            print("Optimization failed.")
            return None, None

    def reset(self):
        """Reset the preference state"""
        self.p_cur = self.p_init


def normalize_simplex(p_mat: np.ndarray) -> np.ndarray:
    """
    Normalize preference states so that they lie in the probability simplex
    :param p_mat: state matrix
    :return: normalized state matrix
    """
    sum_column = np.sum(p_mat, axis=0, keepdims=True)
    p_mat /= sum_column
    return p_mat
