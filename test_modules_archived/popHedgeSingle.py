"""
Analyze an individual with hedge dynamics
"""
import numpy as np


class Population:
    def __init__(self, dim: int, sigma: float, epsilon: float):
        """
        :param dim: dimension of the state, i.e., the number of choices
        :param sigma: combination coefficient in the modified hedge dynamics
        :param epsilon: step size in the hedge dynamics
        """
        # self.p_init = normalize_simplex(np.ones((dim, 1)))  # uniform initial distribution
        self.p_init = normalize_simplex(np.random.rand(dim, 1))
        # self.p_init = normalize_simplex(np.random.normal(loc=1, scale=0.5, size=(dim, 1)))
        self.p_cur = self.p_init
        self.sigma = sigma
        self.epsilon = epsilon
        self.p_traj_data = None

    def modified_hedge_dynamics(self, loss_vec: np.ndarray):
        """
        Implement the modified hedge dynamics
        :param loss_vec: revealed loss vector
        :return: next state, which lies in the probability simplex
        """
        # transient vector; we use the element-wise product
        p_trans = self.p_cur * np.exp(-self.epsilon * loss_vec)
        self.p_cur = self.sigma * normalize_simplex(p_trans) + (1-self.sigma) * self.p_init  # convex combination

    def fixed_response(self, itr_num: int, loss_vec: np.ndarray):
        """
        Analyze the response under a fixed loss_vec
        :param itr_num: number of iterations
        :param loss_vec: revealed loss vector
        """
        dim = loss_vec.shape[0]
        self.p_traj_data = np.zeros((dim, itr_num))
        
        for i in range(itr_num):
            self.p_traj_data[:, i:i+1] = self.p_cur
            self.modified_hedge_dynamics(loss_vec)    


def normalize_simplex(p_mat: np.ndarray) -> np.ndarray:
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
    sigma = 0.2
    epsilon = 1
    itr_num = 100
    pop = Population(dim, sigma, epsilon)
    
    # loss_vec = np.ones((dim, 1))
    loss_vec = np.arange(0, dim).reshape(-1, 1)

    pop.fixed_response(itr_num, loss_vec)
        
    print('Finish running the dynamics')
