import numpy as np

# projection to a box constraint set
def proj_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    projection to a box constraint set
    :param x: point
    :param lb: lower bound
    :param ub: upper bound
    :return y: projection point
    """
    y = x
    sign_id_lower = x < lb
    y[sign_id_lower] = lb[sign_id_lower]
    sign_id_upper = x > ub
    y[sign_id_upper] = ub[sign_id_upper]
    return y