import tensorflow as tf
from .core import TFFMCore
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
import six
from tqdm import tqdm
import numpy as np



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
    """Base class for FM"""

    def init_basemodel(self, rank=2, order=2, input_type='dense', n_epochs=100,
                        loss_function=None, batch_size=-1, reg=0, init_std=0.01,
                        optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
                        log_dir=None, session_config=None, verbose=0):
        core_arguments = {
            'order': order,
            'rank': rank,
            'n_features': None,
            'input_type': input_type,
            'loss_function': loss_function,
            'optimizer': optimizer,
            'reg': reg,
            'init_std': init_std
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
        # TODO: check this
        self.core.set_num_features(X_.shape[1])
        assert self.core.n_features is not None

        if self.core.graph is None:
            self.core.build_graph()
            self.initialize_session()

        # suppose input {0, 1}, but use instead {-1, 1} labels
        # used_y = y_ * 2 - 1
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