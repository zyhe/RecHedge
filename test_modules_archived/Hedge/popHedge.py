"""
Analyze the population with hedge dynamics
"""
import numpy as np


class Population:
    def __init__(self, dim, size_pop, sigma, epsilon):
        """
        :param dim: dimension of the state, i.e., the number of choices
        :param lambda: combination coefficient in the modified hedge dynamics
        :param epsilon: step size in the hedge dynamics
        """
        # self.p_init = normalize_simplex(np.ones((dim, size_pop)))  # uniform initial distribution
        self.p_init = normalize_simplex(np.random.normal(loc=1, scale=0.5, size=(dim, size_pop)))
        self.p_cur = self.p_init
        self.sigma = sigma
        self.epsilon = epsilon

    def modified_hedge_dynamics(self, loss_vec):
        """
        Implement the modified hedge dynamics
        :param loss_vec: revealed loss vector
        :return: next state, which lies in the probability simplex
        """
        # transient vector; we use the element-wise product
        p_trans = self.p_cur * np.exp(-self.epsilon * loss_vec)
        self.p_cur = self.sigma * normalize_simplex(p_trans) + (1-self.sigma) * self.p_init  # convex combination


def normalize_simplex(p_mat):
    """
    Normalize states so that they lie in the probability simplex
    :param p_mat: state matrix
    :return: normalized state matrix
    """
    sum_column = np.sum(p_mat, axis=0, keepdims=True)
    p_mat /= sum_column
    return p_mat


if __name__ == "__main__":
    dim = 5
    size_pop = 10
    sigma = 0.2
    epsilon = 1
    pop = Population(dim, size_pop, sigma, epsilon)
    # loss_vec = np.ones((dim, 1))
    loss_vec = np.arange(0, dim).reshape(-1, 1)
    for i in range(500):
        pop.modified_hedge_dynamics(loss_vec)
    print('Finish running the dynamics')
