"""
## Distribution dynamics

A user following distribution dynamics with a modified hedge structure

"""

import numpy as np
import matplotlib.pyplot as plt


class UserHedge:
    def __init__(self, dim: int, lambda1: float, lambda2: float, epsilon: float):
        """
        :param dim: dimension of the state, i.e., the number of choices
        :param lambda1: combination coefficient in the modified hedge dynamics
        :param lambda2: combination coefficient in the modified hedge dynamics
        :param epsilon: step size/coefficient in the hedge dynamics
        """
        # problem and dynamics parameters
        self.dim = dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epsilon = epsilon

        # preference state
        # self.p_init = normalize_simplex(np.ones((dim, 1)))  # uniform initial distribution
        self.p_init = normalize_simplex(np.random.rand(dim, 1))
        self.p_cur = self.p_init

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
        :param dec: the current loss vector, which serves as the input
        :return: a vector from the softmax function
        """
        max_loss = np.max(-self.epsilon * dec)
        exp_weight = np.exp(-self.epsilon * dec - max_loss)
        exp_weight /= np.sum(exp_weight)
        return exp_weight

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
