"""
Calculate the Jacobian of the softmax function in hedge dynamics
"""
import numpy as np


class Hedge:
    def __init__(self, epsilon: float, dim: int):
        """
        :param epsilon: step size in the hedge dynamics
        :param dim: dimension of the vector
        """
        self.epsilon = epsilon
        self.dim = dim

    def Jacobian(self, t: int, loss: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian of the softmax function in hedge dynamics
        :param t: time index, or the number of iterations
        :param loss: loss vector
        :return: the Jacobian matrix
        :Note: the Jacobian matrix is almost zero for a large t.
               technical reason: numerator t*exp(-t) -> 0, denominator \sum exp(-t) -> N
               intuitive reason: the index of the minimum loss is not sensitive to the small change of the loss vector
               when the elements differ to some degree
        """
        epsilon_t_prod = self.epsilon * t
        # Stabilize exponentials by subtracting the max value of the scaled loss
        max_loss = np.max(-epsilon_t_prod * loss)
        z = np.exp(-epsilon_t_prod * loss - max_loss)
        sum_z = np.sum(z, axis=0)

        diag_z = np.diag(z.ravel())
        jacobian = - epsilon_t_prod * (sum_z * diag_z - z @ z.T) / (sum_z**2)
        return jacobian


def main():
    # set parameters
    epsilon = 0.1
    dim = 5
    loss = np.arange(dim).reshape(-1, 1)

    hedge = Hedge(epsilon, dim)

    for t in [1, 10, 200]:
        print(f'The Jacobian at t={t} is: \n {hedge.Jacobian(t, loss)}')


if __name__ == '__main__':
    main()
