import tensorflow as tf
from .core import TFFMCore
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
import six
from tqdm import tqdm
import numpy as np
import os


def batcher(X_, y_=None, batch_size=-1):
    """Split data to mini-batches.

    Parameters
    ----------
    X_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y_ : np.array or None, shape (n_samples,)
        Target vector relative to X.

    batch_size : int
        Size of batches.
        Use -1 for full-size batches

    Yields
    -------
    ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Same type as input
    ret_y : np.array or None, shape (batch_size,)
    """
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
        yield (ret_x, ret_y)


def batch_to_feeddict(X, y, core):
    """Prepare feed dict for session.run() from mini-batch.
    Convert sparse format into tuple (indices, values, shape) for tf.SparseTensor
    Parameters
    ----------
    X : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Training vector, where batch_size in the number of samples and
        n_features is the number of features.
    y : np.array, shape (batch_size,)
        Target vector relative to X.
    core : TFFMCore
        Core used for extract appropriate placeholders
    Returns
    -------
    fd : dict
        Dict with formatted placeholders
    """
    fd = {}
    if core.input_type == 'dense':
        fd[core.train_x] = X.astype(np.float32)
    else:
        # sparse case
        X_sparse = X.tocoo()
        fd[core.raw_indices] = np.hstack(
            (X_sparse.row[:, np.newaxis], X_sparse.col[:, np.newaxis])
        ).astype(np.int64)
        fd[core.raw_values] = X_sparse.data.astype(np.float32)
        fd[core.raw_shape] = np.array(X_sparse.shape).astype(np.int64)
    if y is not None:
        fd[core.train_y] = y.astype(np.float32)
    return fd




class TFFMBaseModel(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for FM.
    This class implements L2-regularized arbitrary order FM model.

    It supports arbitrary order of interactions and has linear complexity in the
    number of features (a generalization of the approach described in Lemma 3.1
    in the referenced paper, details will be added soon).

    It can handle both dense and sparse input. Only numpy.array and CSR matrix are
    allowed as inputs; any other input format should be explicitly converted.

    Support logging/visualization with TensorBoard.


    Parameters (for initialization)
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

    loss_function : function: (tf.Op, tf.Op) -> tf.Op, default: None
        Loss function.
        Take 2 tf.Ops: outputs and targets and should return tf.Op of loss
        See examples: .core.loss_mse, .core.loss_logistic

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

    def init_basemodel(self, rank=2, order=2, input_type='dense', n_epochs=100,
                        loss_function=None, batch_size=-1, reg=0, init_std=0.01,
                        optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
                        log_dir=None, session_config=None, verbose=0,
                        seed=None):
        core_arguments = {
            'order': order,
            'rank': rank,
            'input_type': input_type,
            'loss_function': loss_function,
            'optimizer': optimizer,
            'reg': reg,
            'init_std': init_std,
            'seed': seed
        }
        self.core = TFFMCore(**core_arguments)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.session_config = session_config
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
        self.session = tf.Session(config=self.session_config, graph=self.core.graph)
        self.session.run(self.core.init_all_vars)

    @abstractmethod
    def preprocess_target(self, target):
        """Prepare target values to use."""

    def fit(self, X_, y_, n_epochs=None, show_progress=False):

        if self.core.n_features is None:
            self.core.set_num_features(X_.shape[1])

        assert self.core.n_features==X_.shape[1], 'Different num of features in initialized graph and input'

        if self.core.graph is None:
            self.core.build_graph()
            self.initialize_session()

        used_y = self.preprocess_target(y_)

        if n_epochs is None:
            n_epochs = self.n_epochs

        # Training cycle
        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):
            if self.verbose > 1:
                print('start epoch: {}'.format(epoch))

            # generate permutation
            perm = np.random.permutation(X_.shape[0])

            # iterate over batches
            for bX, bY in batcher(X_[perm], y_=used_y[perm], batch_size=self.batch_size):
                fd = batch_to_feeddict(bX, bY, core=self.core)
                ops_to_run = [self.core.trainer, self.core.target, self.core.summary_op]
                result = self.session.run(ops_to_run, feed_dict=fd)
                _, batch_target_value, summary_str = result

                if self.verbose > 1:
                    print('target: ' + str(batch_target_value))

                # Write stats
                if self.need_logs:
                    self.summary_writer.add_summary(summary_str, self.steps)
                    self.summary_writer.flush()
                self.steps += 1

    def decision_function(self, X):
        if self.core.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for bX, bY in batcher(X, y_=None, batch_size=self.batch_size):
            fd = batch_to_feeddict(bX, bY, core=self.core)
            output.append(self.session.run(self.core.outputs, feed_dict=fd))
        distances = np.concatenate(output).reshape(-1)
        # TODO: check this reshape
        return distances

    @abstractmethod
    def predict(self, X):
        """Predict target values for X."""

    @property
    def intercept(self):
        """Export bias term from tf.Variable to float."""
        return self.core.b.eval(session=self.session)

    @property
    def weights(self):
        """Export underlying weights from tf.Variables to np.arrays."""
        return [x.eval(session=self.session) for x in self.core.w]

    def save_state(self, path):
        self.core.saver.save(self.session, path)

    def load_state(self, path):
        if self.core.graph is None:
            self.core.build_graph()
            self.initialize_session()
        self.core.saver.restore(self.session, path)

    def destroy(self):
        """Terminate session and destroyes graph."""
        self.session.close()
        self.core.graph = None
