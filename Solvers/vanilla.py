"""
## vanilla algorithm

Follow the idea of performative prediction, i.e., sampling and optimize

"""
import numpy as np
import sys

sys.path.append("..")
from Models.distribution_dynamics import UserHedge


class VanillaAlg:
    def __init__(self, sz: float):
        """
        :param sz: step size
        """
        self.sz = sz

    def itr_update(self, dec_prev: np.ndarray, user: UserHedge, penalty: float, lbd: float) -> np.ndarray:
        """
        Implement the iterative update
        :param dec_prev: previous decision
        :param user: object of the class UserHedge
        :param penalty: current penalty parameter
        :param lbd: lower upper bound on the sum of elements of dec_prev
        """
        constr_vio = max(0, lbd - np.sum(dec_prev))
        grad = user.p_cur - penalty * constr_vio * np.ones_like(dec_prev)
        dec_cur = dec_prev - self.sz * grad
        return dec_cur
