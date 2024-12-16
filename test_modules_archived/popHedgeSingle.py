"""
Analyze an individual with hedge dynamics
"""
import numpy as np


class User:
    def __init__(self, dim: int, sigma: float, epsilon: float):
        """
        :param dim: dimension of the state, i.e., the number of choices
        :param sigma: combination coefficient in the modified hedge dynamics
        :param epsilon: step size in the hedge dynamics
        """
        self.dim = dim
        self.sigma = sigma
        self.epsilon = epsilon
        
        # self.p_init = normalize_simplex(np.ones((dim, 1)))  # uniform initial distribution
        # undesirable due to negative values
        # self.p_init = normalize_simplex(np.random.normal(loc=1, scale=0.5, size=(dim, 1)))
        self.p_init = normalize_simplex(np.random.rand(dim, 1))
        self.p_cur = self.p_init
        self.w_cur = self.p_init
        self.p_traj_data = None

    def modified_hedge_dynamics(self, loss_vec: np.ndarray):
        """
        Implement the modified hedge dynamics
        :param loss_vec: revealed loss vector
        :return: next state, which lies in the probability simplex
        """
        # transient vector
        # rescaling for numerical stability
        max_loss = np.max(-self.epsilon * loss_vec)
        self.w_cur = self.w_cur * np.exp(-self.epsilon * loss_vec - max_loss)
        # p_trans = self.p_cur * np.exp(-self.epsilon * loss_vec)
        self.p_cur = self.sigma * normalize_simplex(self.w_cur) + (1-self.sigma) * self.p_init  # convex combination

    def fixed_response(self, itr_num: int, loss_vec: np.ndarray) -> np.ndarray:
        """
        Analyze the response under a fixed loss_vec
        :param itr_num: number of iterations
        :param loss_vec: revealed loss vector
        :return: the steady-state probability vector
        """
        self.p_traj_data = np.zeros((self.dim, itr_num))
        
        for i in range(itr_num):
            self.p_traj_data[:, i:i+1] = self.p_cur
            self.modified_hedge_dynamics(loss_vec)

        self.reset()
        return self.p_cur

    def reset(self):
        """
        Reset the state to the initial state
        """
        self.w_cur = self.p_init
    
    def compute_jacobian(self, loss_vec: np.ndarray, itr_num: int, epsilon: float = 1e-5) -> np.ndarray:
        """
        Numerically compute the Jacobian of the steady state with respect to the loss vector
        :param loss_vec: loss vector
        :param itr_num: number of iterations
        :param epsilon: small perturbation for numerical differentiation
        :return: Jacobian matrix
        """
        jacobian = np.zeros((self.dim, self.dim))
        
        for i in range(self.dim):
            loss_vec_perturb = loss_vec.copy().astype(np.float64)
            loss_vec_perturb += epsilon
            p_perturb_right = self.fixed_response(itr_num, loss_vec_perturb)
            loss_vec_perturb -= 2 * epsilon
            p_perturb_left = self.fixed_response(itr_num, loss_vec_perturb)

            jacobian[:, i] = (p_perturb_right - p_perturb_left).ravel() / (2 * epsilon)
        return jacobian

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
    np.random.seed(0)
    dim = 5
    sigma = 0.4
    epsilon = 1
    itr_num = 50
    user = User(dim, sigma, epsilon)
    
    # loss_vec = np.ones((dim, 1))
    loss_vec = np.arange(0, dim).reshape(-1, 1)
    _ = user.fixed_response(itr_num, loss_vec)

    # find the index of the minimum element and construct the corresponding array
    min_id_vec = (loss_vec == np.min(loss_vec)).astype(int)
    p_final_theory = user.sigma * min_id_vec + (1-user.sigma)*user.p_init
    p_final_empirical = user.p_cur[:, -1:]
    print(f'The difference in terms of the final distribution is {np.linalg.norm(p_final_theory - p_final_empirical)}')
    
    jacobian = user.compute_jacobian(loss_vec, itr_num)
    print(f'The Jacobian matrix is: \n {jacobian}')
    
    print('Finish running the dynamics')
