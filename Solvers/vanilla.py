"""
## vanilla algorithm

Follow the idea of performative prediction, i.e., sampling and optimize

"""
import numpy as np


class VanillaAlg:
    def __init__(self, sz: float, rho: float, decay_fac: float):
        """
        :param sz: step size
        :param rho: penalty parameter
        :param decay_fac: decay factor
        """
        self.sz = sz
        self.rho = rho
        self.decay_fac = decay_fac

    def itr_update(self, dec_prev) -> np.ndarray:
        """
        Implement the iterative update
        :param dec_prev: previous decision
        """
        grad = 0
        dec_cur = dec_prev - self.sz * grad
        print(dec_cur)

        return dec_cur
