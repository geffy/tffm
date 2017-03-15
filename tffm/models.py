"""Implementation of an arbitrary order Factorization Machines."""

import os
import numpy as np
import tensorflow as tf
import shutil
from tqdm import tqdm


from .core import TFFMCore
from .base import TFFMBaseModel
from .utils import loss_logistic, loss_mse, sigmoid



class TFFMClassifier(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with logistic
    loss and gradient-based optimization.

    Only binary classification with 0/1 labels supported.

    See TFFMBaseModel and TFFMCore docs for details about parameters.
    """

    def __init__(self, **init_params):
        assert 'loss_function' not in init_params
        init_params['loss_function'] = loss_logistic
        self.init_basemodel(**init_params)

    def preprocess_target(self, y_):
        # suppose input {0, 1}, but internally will use {-1, 1} labels instead
        assert(set(y_) == set([0, 1]))
        return y_ * 2 - 1

    def predict(self, X):
        """Predict using the FM model

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Returns predicted values.
        """
        raw_output = self.decision_function(X)
        predictions = (raw_output > 0).astype(int)
        return predictions

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all 2 classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        probs : array-like, shape = [n_samples, 2]
            Returns the probability of the sample for each class in the model.
        """
        outputs = self.decision_function(X)
        probs_positive = sigmoid(outputs)
        probs_negative = 1 - probs_positive
        probs = np.vstack((probs_negative.T, probs_positive.T))
        return probs.T


class TFFMRegressor(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with MSE
    loss and gradient-based optimization.

    See TFFMBaseModel and TFFMCore docs for details about parameters.
    """

    def __init__(self, **init_params):
        assert 'loss_function' not in init_params
        init_params['loss_function'] = loss_mse
        self.init_basemodel(**init_params)

    def preprocess_target(self, y_):
        return y_

    def predict(self, X):
        """Predict using the FM model

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Returns predicted values.
        """
        predictions = self.decision_function(X)
        return predictions
