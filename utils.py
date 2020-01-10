# Author: Alex DELALANDE
# Date:   10 Jan 2020

# Utilities for semi-discrete OT


import numpy as np
from scipy.sparse import csr_matrix

from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import PowerDiagram


# utilities
def make_square(box=[0, 0, 1, 1]):
    """
    Constructs a square domain with uniform measure (source measure 'rho').
    To be passed to the 'newton_ot' and 'PowerDiagram' functions.
    Args:
        box (list): coordinates of the bottom-left and top-right corners
    Returns:
        domain (pysdot.domain_types): domain
    """
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1]], [box[2], box[3]])
    return domain

def laguerre_areas(domain, Y, psi, der=False):
    """
    Computes the areas of the Laguerre cells intersected with the domain.
    Args:
        domain (pysdot.domain_types): domain of the (continuous)
                                      source measure
        Y (np.array): points of the (discrete) target measure
        psi (np.array or list): Kantorovich potentials
        der (bool): wether or not return the Jacobian of the areas
                    w.r.t. psi
    Returns:
        pd.integrals() (list): list of areas of Laguerre cells
    """
    pd = PowerDiagram(Y, -psi, domain)
    if der:
        N = len(psi)
        mvs = pd.der_integrals_wrt_weights()
        return mvs.v_values, csr_matrix((-mvs.m_values, mvs.m_columns, mvs.m_offsets), shape=(N, N))
    else:
        return pd.integrals()