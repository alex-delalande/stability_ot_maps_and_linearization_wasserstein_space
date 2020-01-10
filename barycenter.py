# Author: Alex DELALANDE
# Date:   10 Jan 2020

# Barycenter approximation with Monge's embedding


import numpy as np

from optimal_transport import semi_discrete_ot

class monge_barycenter:
    """
    Class for computing point clouds barycenters with Monge's embeddings
    """

    def __init__(self, point_clouds, grid_size=25, rescaling=True, *args, **kwargs):
        """
        Args:
            point_clouds (dict): dictionary of point clouds
            grid_size (int): discretization parameter for the transport plans
                             (m in the paper)
            rescaling (bool): rescale or not the coordinates of point clouds
                              to fit in [0, 1]² to make computations easier
                              (rescaling undone when returning transport plans)
        """
        self.point_clouds = point_clouds
        self.grid_size = grid_size
        self.rescaling = rescaling
        # compute tranport plans of point clouds
        self.ot_clouds = semi_discrete_ot(grid_size=grid_size)
        self.ot_clouds.fit_transport_plans(self.point_clouds, rescaling=self.rescaling)


    def fit(self, barycenter_coeffs,  *args, **kwargs):
        """
        Compute barycentric transport map and returns the support of the
        associated approximate barycenter.
        Args:
            barycenter_coeffs (list): list of barycentric coefficients
                                      (same length as 'point_clouds')
        Returns:
            barycenter (array): coordinates of the points in the support
                                of the approximate barycenter
        """
        # Compute transport plan of barycenters as barycenter of transport plans
        barycenter_coeffs = np.array(barycenter_coeffs).reshape((len(self.point_clouds), 1))
        T_barycenter = np.sum(barycenter_coeffs * self.ot_clouds.transport_plans,\
                              axis=0).reshape((-1, 2))
        # Get barycenter support with the image of its transport plan
        self.barycenter = np.unique(T_barycenter, axis=0)
