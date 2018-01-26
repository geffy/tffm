"""Implementation of an arbitrary order Factorization Machines."""

import numpy as np
from .base import TFFMBaseModel
from .utils import loss_logistic, loss_mse,  sigmoid



class TFFMClassifier(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with logistic
    loss and gradient-based optimization.

    Only binary classification with 0/1 labels supported.

    See TFFMBaseModel and TFFMCore docs for details about parameters.
    """

    def __init__(self, **init_params):

        assert 'loss_function' not in init_params, """Parameter 'loss_function' is
        not supported for TFFMClassifier. For custom loss function, extend the
        base class TFFMBaseModel."""

        init_params['loss_function'] = loss_logistic
        self.init_basemodel(**init_params)

    def _preprocess_sample_weights(self, sample_weight, pos_class_weight, used_y):
        assert sample_weight is None or pos_class_weight is None, "sample_weight and pos_class_weight are mutually exclusive parameters"
        used_w = np.ones_like(used_y)
        if sample_weight is None and pos_class_weight is None:
            return used_w
        if type(pos_class_weight) == float:
            used_w[used_y > 0] = pos_class_weight
        elif sample_weight == "balanced":
            pos_rate = np.mean(used_y > 0)
            neg_rate = 1 - pos_rate
            used_w[used_y > 0] = neg_rate / pos_rate
            used_w[used_y < 0] = 1.0
            return used_w
        elif type(sample_weight) == np.ndarray and len(sample_weight.shape)==1:
            used_w = sample_weight
        else:
            raise ValueError("Unexpected type for sample_weight or pos_class_weight parameters.")

        return used_w


    def fit(self, X, y, sample_weight=None, pos_class_weight=None, n_epochs=None, show_progress=False):
        # preprocess Y: suppose input {0, 1}, but internally will use {-1, 1} labels instead
        if not (set(y) == set([0, 1])):
            raise ValueError("Input labels must be in set {0,1}.")
        used_y = y * 2 - 1
        if sample_weight is not None:
            self.sample_weight = sample_weight
        if pos_class_weight is not None:
            self.pos_class_weight = pos_class_weight
        used_w = self._preprocess_sample_weights(self.sample_weight, self.pos_class_weight, used_y)
        self._fit(X_=X, y_=used_y, w_=used_w, n_epochs=n_epochs, show_progress=show_progress)

    def predict(self, X, pred_batch_size=None):
        """Predict using the FM model

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.
        pred_batch_size : int batch size for prediction (default None)

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Returns predicted values.
        """
        raw_output = self.decision_function(X, pred_batch_size)
        predictions = (raw_output > 0).astype(int)
        return predictions

    def predict_proba(self, X, pred_batch_size=None):
        """Probability estimates.

        The returned estimates for all 2 classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        pred_batch_size : int batch size for prediction (default None)

        Returns
        -------
        probs : array-like, shape = [n_samples, 2]
            Returns the probability of the sample for each class in the model.
        """
        outputs = self.decision_function(X, pred_batch_size)
        probs_positive = sigmoid(outputs)
        probs_negative = 1 - probs_positive
        probs = np.vstack((probs_negative.T, probs_positive.T))
        return probs.T


class TFFMRegressor(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with MSE
    loss and gradient-based optimization.

    Custom loss functions are not supported, mean squared error is always
    used. Any loss function provided in parameters will be overwritten.

    See TFFMBaseModel and TFFMCore docs for details about parameters.
    """

    def __init__(self, **init_params):

        assert 'loss_function' not in init_params, """Parameter 'loss_function' is
        not supported for TFFMRegressor. For custom loss function, extend the
        base class TFFMBaseModel."""
        
        init_params['loss_function'] = loss_mse
        self.init_basemodel(**init_params)

    def fit(self, X, y, sample_weight=None, n_epochs=None, show_progress=False):
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight
        self._fit(X_=X, y_=y, w_=sample_weight, n_epochs=n_epochs, show_progress=show_progress)

    def predict(self, X, pred_batch_size=None):
        """Predict using the FM model

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.
        pred_batch_size : int batch size for prediction (default None)

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Returns predicted values.
        """
        predictions = self.decision_function(X, pred_batch_size)
        return predictions
