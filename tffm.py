"""Implementation of an arbitrary order Factorization Machines."""


import numpy as np
import tensorflow as tf
from tqdm import tqdm
import core

from core import TFFMCore
import sklearn
from sklearn.base import BaseEstimator
import os
import shutil


class TFFMClassifier(BaseEstimator):
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

    def __init__(
        self,
        rank,
        order=2,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
        batch_size=-1,
        n_epochs=100,
        reg=0,
        init_std=0.01,
        input_type='dense',
        log_dir=None, verbose=0
    ):
        self.core = TFFMCore(
            order=order,
            rank=rank,
            n_features=None,
            input_type=input_type,
            optimizer=optimizer,
            reg=reg,
            init_std=init_std)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.verbose = verbose
        self.steps = 0

    def initialize_session(self):
        """Start computational session on builded graph.

        Initialize summary logger (if needed).
        """
        if self.core.graph is None:
            raise 'Graph not found. Try call .core.build_graph() before .initialize_session()'
        if self.need_logs:
            self.summary_writer = tf.train.SummaryWriter(
                self.log_dir,
                self.core.graph)
            if self.verbose > 0:
                print('Initialize logs, use: \ntensorboard --logdir={}'.format(
                    os.path.abspath(self.log_dir)))
        self.session = tf.Session(graph=self.core.graph)
        self.session.run(self.core.init_all_vars)

    def destroy(self):
        """Terminate session and destroyes graph."""
        self.session.close()
        self.core.graph = None

    def batcher(self, X_, y_=None):
        """Split data to mini-batches.

        Parameters
        ----------
        X_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y_ : np.array or None, shape (n_samples,)
            Target vector relative to X.

        Yields
        -------
        ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
            Same type as input
        ret_y : np.array or None, shape (batch_size,)
        """
        if self.batch_size == -1:
            batch_size = X_.shape[0]
        else:
            batch_size = self.batch_size

        n_samples = X_.shape[0]
        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            ret_x = X_[i:upper_bound]
            ret_y = None
            if y_ is not None:
                ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

    def batch_to_feeddict(self, X, y):
        """Prepare feed dict for session.run() from mini-batch.

        Convert sparse format into tuple (indices, values, shape) for ts.SparseTensor

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
            Training vector, where batch_size in the number of samples and
            n_features is the number of features.

        y : np.array, shape (batch_size,)
            Target vector relative to X.

        Returns
        -------
        fd : dict
            Dict with formatted placeholders
        """
        fd = {}
        if self.core.input_type == 'dense':
            fd[self.core.train_x] = X.astype(np.float32)
        else:
            # sparse case
            X_sp = X.tocoo()
            fd[self.core.raw_indices] = np.hstack((
                X_sp.row[:, np.newaxis],
                X_sp.col[:, np.newaxis])).astype(np.int64)
            fd[self.core.raw_values] = X_sp.data.astype(np.float32)
            fd[self.core.raw_shape] = np.array(X_sp.shape).astype(np.int64)
        if y is not None:
            fd[self.core.train_y] = y.astype(np.float32)
        return fd

    def fit(self, X_, y_, n_epochs=None, show_progress=False):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y_ : np.array, shape (n_samples,)
            Target vector relative to X.

        n_epochs : int or None, default: None
            Number of learning epochs. If is None -- self.n_epochs will be used

        show_progress : bool, default: False
            Specifies if a progress bar should be printed.
        """
        self.core.set_num_features(X_.shape[1])
        assert self.core.n_features is not None
        if self.core.graph is None:
            self.core.build_graph()
            self.initialize_session()
        # suppose input {0, 1}, but use instead {-1, 1} labels
        used_y = y_ * 2 - 1

        if n_epochs is None:
            n_epochs = self.n_epochs

        # Training cycle
        for epoch in tqdm(
            range(n_epochs), unit='epoch', disable=(not show_progress)
        ):
            if self.verbose > 1:
                print 'start epoch: {}'.format(epoch)

            # generate permutation
            perm = np.random.permutation(X_.shape[0])

            # iterate over batches
            for i, (bX, bY) in enumerate(self.batcher(X_[perm], used_y[perm])):
                fd = self.batch_to_feeddict(bX, bY)
                _, batch_target_value, summary_str = self.session.run(
                    [self.core.trainer, self.core.target, self.core.summary_op],
                    feed_dict=fd)

                if self.verbose > 1:
                    print ' -> batch: {}, target: {},'.format(
                        i,
                        batch_target_value)

                # Write stats
                if self.need_logs:
                    self.summary_writer.add_summary(summary_str, self.steps)
                    self.summary_writer.flush()
                self.steps += 1

    def decision_function(self, X):
        """Decision function of the FM model.

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        distances : array, shape = (n_samples,)
            Returns predicted values.
        """
        if self.core.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for (bX, bY) in self.batcher(X):
            fd = self.batch_to_feeddict(bX, bY)
            output.append(self.session.run(self.core.outputs, feed_dict=fd))
        distances = np.concatenate(output).reshape(-1)
        return distances

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
        if self.core.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for (bX, bY) in self.batcher(X):
            fd = self.batch_to_feeddict(bX, bY)
            output.append(self.session.run(self.core.probs, feed_dict=fd))
        probs_positive = np.concatenate(output)
        probs_negative = 1 - probs_positive
        probs = np.concatenate((probs_negative, probs_positive), axis=1)
        return probs

    @property
    def intercept(self):
        """Export bias term from tf.Variable to float."""
        return self.core.b.eval(session=self.session)

    @property
    def weights(self):
        """Export underlying weights from tf.Variables to np.arrays."""
        return [x.eval(session=self.session) for x in self.core.w]

    def save_state(self, path):
        """Save current session to file.

        Parameters
        ----------
        path : str
            Destination file
        """
        self.core.saver.save(self.session, path)

    def load_state(self, path):
        """Restore session state from file.

        Parameters
        ----------
        path : str
            Restoring file
        """
        if self.core.graph is None:
            self.core.build_graph()
            self.initialize_session()
        self.core.saver.restore(self.session, path)
