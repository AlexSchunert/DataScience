from numpy import ndarray, empty
from dataclasses import dataclass

@dataclass
class GPPosterior:
    """
    Contains the posterior of a gp given by represented weights and predictive covariance
    """

    repr_weights: ndarray = empty,
    predictive_cov: ndarray = empty

    def __int__(self,
                repr_weights,
                predictive_cov):
        """
        Inits GPPosterior

        :param repr_weights: Representer weights determined during conditioning
        :type repr_weights: ndarray
        :param predictive_cov: Predictive covariance determined during conditioning
        :type predictive_cov: ndarray

        :return: ---
        :rtype: None
        """
        self.repr_weights = repr_weights
        self.predictive_cov = predictive_cov