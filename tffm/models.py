"""Implementation of an arbitrary order Factorization Machines."""

import os
import numpy as np
import tensorflow as tf
import shutil
from tqdm import tqdm


from .core import TFFMCore
from .base import TFFMBaseModel
from .utils import loss_logistic, loss_mse



class TFFMClassifier(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with logistic
    loss and gradient-based optimization.

    Only binary classification with 0/1 labels supported.

    See TFFMBaseModel docs for details about parameters.
    """

    def __init__(self, rank=2, order=2, input_type='dense', n_epochs=100,
                optimizer=tf.train.AdamOptimizer(learning_rate=0.1), reg=0,
                batch_size=-1, init_std=0.01, log_dir=None, verbose=0,
                session_config=None, seed=None):
        init_params = {
            'rank': rank,
            'order': order,
            'input_type': input_type,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'reg': reg,
            'init_std': init_std,
            'optimizer': optimizer,
            'log_dir': log_dir,
            'loss_function': loss_logistic,
            'verbose': verbose,
            'seed': seed
        }
        self.init_basemodel(**init_params)

    def preprocess_target(self, y_):
        # suppose input {0, 1}, but use instead {-1, 1} labels
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
        probs_positive = utils.sigmoid(outputs)
        probs_negative = 1 - probs_positive
        probs = np.concatenate((probs_negative, probs_positive), axis=1)
        return probs


class TFFMRegressor(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with MSE
    loss and gradient-based optimization.

    See TFFMBaseModel docs for details about parameters.
    """

    def __init__(self, rank=2, order=2, input_type='dense', n_epochs=100,
                optimizer=tf.train.AdamOptimizer(learning_rate=0.1), reg=0,
                batch_size=-1, init_std=0.01, log_dir=None, verbose=0,
                session_config=None, seed=None):
        init_params = {
            'rank': rank,
            'order': order,
            'input_type': input_type,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'reg': reg,
            'init_std': init_std,
            'optimizer': optimizer,
            'log_dir': log_dir,
            'loss_function': loss_mse,
            'verbose': verbose,
            'seed': seed
        }
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
