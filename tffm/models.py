"""Implementation of an arbitrary order Factorization Machines."""

import os
import numpy as np
import tensorflow as tf
import shutil
from tqdm import tqdm


from .core import TFFMCore, loss_logistic, loss_mse
from .base import TFFMBaseModel



class TFFMClassifier(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with logistic
    loss and gradient-based optimization.

    It supports arbitrary order of interactions and has linear complexity in the
    number of features (a generalization of the approach described in Lemma 3.1
    in the referenced paper, details will be added soon).

    It can handle both dense and sparse input. Only numpy.array and CSR matrix are
    allowed as inputs; any other input format should be explicitly converted.

    Support logging/visualization with TensorBoard.

    Only binary classification with 0/1 labels supported.

    Parameters
    ----------
    rank : int
        Number of factors in low-rank appoximation.
        This value is shared across different orders of interaction.

    order : int, default: 2
        Order of corresponding polynomial model.
        All interaction from bias and linear to order will be included.

    optimizer : tf.train.Optimizer, default: AdamOptimizer(learning_rate=0.1)
        Optimization method used for training

    batch_size : int, default: -1
        Number of samples in mini-batches. Shuffled every epoch.
        Use -1 for full gradient (whole training set in each batch).

    n_epoch : int, default: 100
        Default number of epoches.
        It can be overrived by explicitly provided value in fit() method.

    reg : float, default: 0
        Strength of L2 regularization

    init_std : float, default: 0.01
        Amplitude of random initialization

    input_type : str, 'dense' or 'sparse', default: 'dense'
        Type of input data. Only numpy.array allowed for 'dense' and
        scipy.sparse.csr_matrix for 'sparse'. This affects construction of
        computational graph and cannot be changed during training/testing.

    log_dir : str or None, default: None
        Path for storing model stats during training. Used only if is not None.
        WARNING: If such directory already exists, it will be removed!
        You can use TensorBoard to visualize the stats:
        `tensorboard --logdir={log_dir}`

    session_config : tf.ConfigProto or None, default: None
        Additional setting passed to tf.Session object.
        Useful for CPU/GPU switching.
        `tf.ConfigProto(device_count = {'GPU': 0})` will disable GPU (if enabled)


    verbose : int, default: 0
        Level of verbosity.
        Set 1 for tensorboard info only and 2 for additional stats every epoch.

    Attributes
    ----------
    core : TFFMCore or None
        Computational graph with internal utils.
        Will be initialized during first call .fit()

    session : tf.Session or None
        Current execution session or None.
        Should be explicitly terminated via calling destroy() method.

    steps : int
        Counter of passed lerning epochs, used as step number for writing stats

    n_features : int
        Number of features used in this dataset.
        Inferred during the first call of fit() method.

    intercept : float, shape: [1]
        Intercept (bias) term.

    weights : array of np.array, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    Notes
    -----
    You should explicitly call destroy() method to release resources.
    Parameter rank is shared across all orders of interactions (except bias and
    linear parts).
    tf.sparse_reorder doesn't requied since COO format is lexigraphical ordered.

    References
    ----------
    Steffen Rendle, Factorization Machines
        http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """

    def __init__(self, rank=2, order=2, input_type='dense', n_epochs=100,
                optimizer=tf.train.AdamOptimizer(learning_rate=0.1), reg=0,
                batch_size=-1, init_std=0.01, log_dir=None, verbose=0,
                session_config=None):
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
            'loss_function': loss_logistic
        }
        self.init_basemodel(**init_params)

    def preprocess_target(self, y_):
        # suppose input {0, 1}, but use instead {-1, 1} labels
        assert(set(y_)==set([0, 1]))
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
        # TODO: preprocess classes
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