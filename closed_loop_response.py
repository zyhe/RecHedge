"""
Analyze the closed-loop response of the user and the algorithm
We address hedge dynamics and an equality constraint on budget
"""
import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime
from pathlib import Path
import yaml
import cProfile  # for profile analysis
import pstats

import warnings
warnings.simplefilter("error", RuntimeWarning)

from Models.distribution_dynamics import UserHedge
from Solvers.vanilla import VanillaAlg
from Solvers.composite import CompositeAlg

# Configure font settings
plt.rcParams.update({
    "font.family": "Times New Roman",  # another option "serif"
    "font.size": 14,
    "mathtext.fontset": "cm",  # Computer Modern for math
})


class ClosedLoopResponse:
    def __init__(self, file_path: str = 'Config/params.yaml'):
        self.file_path = file_path
        self.params = self._load_parameters()

        # parameters related to the distribution dynamics
        self.lambda1 = self.params['dynamics']['lambda1']
        self.lambda2 = self.params['dynamics']['lambda1']
        self.epsilon = self.params['dynamics']['epsilon']

        # parameters related to the algorithm
        self.sz = self.params['algorithm']['sz']
        self.num_itr = int(float(self.params['algorithm']['num_itr']))
        # self.penalty_coeff = self.params['algorithm']['penalty_coeff']
        # self.penalty_inc_factor = self.params['algorithm']['penalty_inc_factor']

        # parameters related to the problem
        self.dim = self.params['problem']['dim']
        # self.bd_dec = self.params['problem']['bd_dec']
        self.budget = self.params['problem']['budget']

        # initialize the user distribution and the algorithm
        self.user = UserHedge(self.dim, self.lambda1, self.lambda2, self.epsilon, self.budget)
        self.alg = None  # type of the solver
        self.alg_name = {0: 'vanilla', 1: 'composite'}

        # store results
        self.pref_data = np.zeros((2, self.dim, self.num_itr))
        self.dec_data = np.zeros((2, self.dim, self.num_itr))
        self.utility_data = np.zeros((2, self.num_itr))
        self.constraint_vio_data = np.zeros((2, self.num_itr))

    def _load_parameters(self) -> dict:
        """
        Load configuration parameters from a YAML file.
        :param file_path: Path to the YAML configuration file.
        :return: Dictionary of loaded parameters.
        """
        with Path(self.file_path).open('r') as file:
            return yaml.safe_load(file)

    def _select_solver(self, mode):
        """Select the solver based on the given mode."""
        if mode == "vanilla":
            return VanillaAlg(self.sz)
        elif mode == "composite":
            return CompositeAlg(self.sz)

    def feedback_response(self, index):
        """
        Implement the response when the algorithm is interconnected with the distribution dynamics
        :param index: index of the problem, 0 for vanilla, and 1 for composite
        """
        # initial decision
        dec = self.budget / self.dim * np.ones((self.dim, 1))  # equal weight to each element
        # dec = self.budget * (self.user.p_init == np.max(self.user.p_init)).astype(int)  # greedy initial decision
        # penalty_cur = self.penalty_coeff

        for i in range(self.num_itr):
            # penalty_cur *= self.penalty_inc_factor
            p_cur = self.user.per_step_dynamics(dec)
            dec = self.alg.itr_update(dec, self.user, self.budget)

            # store results
            self.pref_data[index, :, i:i+1] = p_cur
            self.dec_data[index, :, i:i+1] = dec
            self.utility_data[index, i], self.constraint_vio_data[index, i] = self.evaluate_perf(dec)

        print(f'The {self.alg_name[index]} algorithm is finished.')

    def evaluate_perf(self, dec: np.ndarray) -> tuple[float, float]:
        """Evaluate the performance in terms of the objective and constraint satisfaction."""
        utility = (self.user.p_cur.T @ dec).item()
        constraint_violation = (np.sum(dec) - self.budget)**2
        # constraint_violation = max(0, self.lbd - np.sum(dec))  # if positive, then the constraint is violated
        return utility, constraint_violation

    def _visualize(self):
        """Plot utility and constraint violation over iterations."""
        self._plot_metric(self.utility_data, "Loss")  # utility
        # self._plot_metric(self.constraint_vio_data, "Constraint Violation")  # constraint violation
        plt.show()

    def _plot_metric(self, data: np.ndarray, ylabel: str):
        """Plot the evolution of a specific metric."""
        plt.figure()
        plt.plot(np.arange(self.num_itr), data[0], linewidth=2, label='vanilla')
        plt.plot(np.arange(self.num_itr), data[1], linewidth=2, label='composite')
        plt.legend(fontsize=16, loc='lower right')
        plt.xlabel('Number of Iterations', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14, direction='in', length=4, width=0.5)
        plt.tick_params(axis='both', which='minor', labelsize=10, direction='in', length=2, width=0.5)  # Minor ticks
        plt.minorticks_on()
        plt.locator_params(axis='both', nbins=7)
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.show(block=False)

    def execute(self):
        """
        Executes full closed-loop responses with different algorithms.
        """
        # Run the vanilla algorithm
        self.alg = self._select_solver("vanilla")
        self.feedback_response(index=0)

        # Run the composite algorithm
        self.alg = self._select_solver("composite")
        self.feedback_response(index=1)

        print(f'The utility obtained by the naive solution is {self.user.naive_dec_utility()}')

        self._visualize()
        print('Finish the program.')

    @staticmethod
    def profile_execution():
        """
        Profiles the execution of the main analysis process.
        """
        cProfile.run("ClosedLoopResponse().execute()", "main_stats")
        p = pstats.Stats("main_stats")
        p.sort_stats("cumulative").print_stats(50)


def main():
    np.random.seed(5)
    response_runner = ClosedLoopResponse(file_path='./Config/params.yaml')
    response_runner.execute()
    # response_runner.profile_execution()


if __name__ == "__main__":
    main()
