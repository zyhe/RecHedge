import numpy as np
from scipy.optimize import minimize

class User:
    def __init__(self, dim, lambda1, lambda2, epsilon, p_init):
        self.dim = dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epsilon = epsilon
        self.p_init = p_init

    def softmax_vec(self, dec: np.ndarray) -> np.ndarray:
        dec_shift = dec - np.max(dec)
        exp_weight = np.exp(self.epsilon * dec_shift)
        return exp_weight / np.sum(exp_weight)

    def steady_state(self, dec: np.ndarray) -> np.ndarray:
        p_ss = ((self.lambda2 * self.softmax_vec(dec) + (1 - self.lambda1 - self.lambda2) * self.p_init)
                / (1 - self.lambda1))
        return p_ss

    def objective_function(self, dec: np.ndarray) -> float:
        steady_state = self.steady_state(dec)
        return -np.dot(dec, steady_state)  # Negative because we are maximizing

    def constraint_sum(self, dec: np.ndarray) -> float:
        return np.sum(dec) - 1

    def maximize_objective(self):
        """
        Optimize the objective function (i.e., the inner product) via scipy.optimize.minimize
        :return: optimal decision and optimal value
        """
        # Initial guess
        x0 = np.ones(self.dim) / self.dim

        # Constraints
        constraints = {'type': 'eq', 'fun': self.constraint_sum}

        # Bounds for decision variables
        bounds = [(0, 1) for _ in range(self.dim)]

        # Optimize
        result = minimize(self.objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimal_dec = result.x
            optimal_value = -result.fun  # Negate to get the original objective value
            print("Optimal decision vector:", optimal_dec)
            print("Optimal objective value:", optimal_value)
            return optimal_dec, optimal_value
        else:
            print("Optimization failed.")
            return None, None

# Example usage
dim = 3
lambda1 = 0.5
lambda2 = 0.3
epsilon = 1.0
p_init = np.array([0.2, 0.3, 0.5])

user_instance = User(dim, lambda1, lambda2, epsilon, p_init)
optimal_dec, optimal_value = user_instance.maximize_objective()
print('finish')