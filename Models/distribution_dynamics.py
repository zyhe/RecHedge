"""
## Distribution dynamics

A user following distribution dynamics with a modified hedge structure

"""

import numpy as np
import matplotlib.pyplot as plt


class UserHedge:
    def __init__(self, dim: int, sigma: float, epsilon: float):
        """
        :param dim: dimension of the state, i.e., the number of choices
        :param sigma: combination coefficient in the modified hedge dynamics
        :param epsilon: step size in the hedge dynamics
        """
        # problem and dynamics parameters
        self.dim = dim
        self.sigma = sigma
        self.epsilon = epsilon

        # preference state
        # self.p_init = normalize_simplex(np.ones((dim, 1)))  # uniform initial distribution
        self.p_init = normalize_simplex(np.random.rand(dim, 1))
        self.p_cur = self.p_init
        self.p_traj_data = None

    def per_step_dynamics(self, dec: np.ndarray) -> np.ndarray:
        """
        Implement the modified hedge dynamics at every time step
        :param: dec: the current loss vector, which serves as the input
        :return: the preference state at the next time step
        """
        # transient vector
        p_trans = self.p_cur * np.exp(-self.epsilon * dec)
        self.p_cur = self.sigma * normalize_simplex(p_trans) + (1 - self.sigma) * self.p_init  # convex combination
        return self.p_cur


def normalize_simplex(p_mat: np.ndarray) -> np.ndarray:
    """
    Normalize preference states so that they lie in the probability simplex
    :param p_mat: state matrix
    :return: normalized state matrix
    """
    sum_column = np.sum(p_mat, axis=0, keepdims=True)
    p_mat /= sum_column
    return p_mat
