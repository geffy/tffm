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
    batch_size : int, default: -1
        Number of samples in mini-batches. Shuffled every epoch.
        Use -1 for full gradient (whole training set in each batch).

    n_epoch : int, default: 100
        Default number of epoches.
        It can be overrived by explicitly provided value in fit() method.

    log_dir : str or None, default: None
        Path for storing model stats during training. Used only if is not None.
        WARNING: If such directory already exists, it will be removed!
        You can use TensorBoard to visualize the stats:
        `tensorboard --logdir={log_dir}`

    session_config : tf.ConfigProto or None, default: None
        Additional setting passed to tf.Session object.
        Useful for CPU/GPU switching, setting number of threads and so on,
        `tf.ConfigProto(device_count = {'GPU': 0})` will disable GPU (if enabled)

    verbose : int, default: 0
        Level of verbosity.
        Set 1 for tensorboard info only and 2 for additional stats every epoch.

    kwargs : dict, default: {}
        Arguments for TFFMCore constructor.
        See TFFMCore's doc for details.

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
    See TFFMCore's doc for details.
    """


    def init_basemodel(self, n_epochs=100, batch_size=-1, log_dir=None, session_config=None, verbose=0, seed=None, **core_arguments):
        core_arguments['seed'] = seed
        self.core = TFFMCore(**core_arguments)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.session_config = session_config
        self.verbose = verbose
        self.steps = 0
        self.seed = seed

    def initialize_session(self):
        """Start computational session on builded graph.
        Initialize summary logger (if needed).
        """
        if self.core.graph is None:
            raise 'Graph not found. Try call .core.build_graph() before .initialize_session()'
        if self.need_logs:
            self.summary_writer = tf.summary.FileWriter(self.log_dir, self.core.graph)
            if self.verbose > 0:
                full_log_path = os.path.abspath(self.log_dir)
                print('Initialize logs, use: \ntensorboard --logdir={}'.format(full_log_path))
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

        # For reproducible results
        if self.seed:
            np.random.seed(self.seed)

        # Training cycle
        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):
            # generate permutation
            perm = np.random.permutation(X_.shape[0])
            epoch_loss = []
            # iterate over batches
            for bX, bY in batcher(X_[perm], y_=used_y[perm], batch_size=self.batch_size):
                fd = batch_to_feeddict(bX, bY, core=self.core)
                ops_to_run = [self.core.trainer, self.core.target, self.core.summary_op]
                result = self.session.run(ops_to_run, feed_dict=fd)
                _, batch_target_value, summary_str = result
                epoch_loss.append(batch_target_value)
                # write stats 
                if self.need_logs:
                    self.summary_writer.add_summary(summary_str, self.steps)
                    self.summary_writer.flush()
                self.steps += 1
            if self.verbose > 1:
                    print('[epoch {}]: mean target value: {}'.format(epoch, np.mean(epoch_loss)))

    def decision_function(self, X):
        if self.core.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for bX, bY in batcher(X, y_=None, batch_size=self.batch_size):
            fd = batch_to_feeddict(bX, bY, core=self.core)
            output.append(self.session.run(self.core.outputs, feed_dict=fd))
        distances = np.concatenate(output).reshape(-1)
        # WARN: be carefull with this reshape in case of multiclass
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
        """Terminates session and destroyes graph."""
        self.session.close()
        self.core.graph = None
