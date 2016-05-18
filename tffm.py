"""Implementation of an arbitrary order Factorization Machines."""

# Author: Mikhail Trofimov <mikhail.trofimov@phystech.edu>
# License: MIT

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
import core
import math
import sklearn
from sklearn.base import BaseEstimator
import os
import shutil


class TFFMClassifier(BaseEstimator):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with logistic
    loss and gradient-based optimization.
    It supports arbitrary order of interactions and has linear complexity in in
     number of features (as described in Lemma 3.1 in the referenced paper).
    It can handle both dense and sparse input. Only numpy.array and CSR matrix
    are allowed; any other input format should be explicitly converted.
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

    fit_intercept : bool, default: True
        Whether the intercept should be estimated or not.

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
    graph : tf.Graph or None
        Initialize computational graph or None

    session : tf.Session or None
        Current execution session or None.
        Should be explicitly terminated via calling destroy() method.

    trainer : tf.Op
        TensorFlow operation node to perform learning on single batch

    steps : int
        Counter of passed lerning epochs, used as step number for writing stats

    n_features : int
        Number of features used in this dataset.
        Inferred during the first call of fit() method.

    b : tf.Variable, shape: [1]
        Bias term.

    w : array of tf.Variable, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    Notes
    -----
    You should explicitly call destroy() method to release resources
    Parameter rank is shared across all orders of interactions
    (except bias and linear parts).
    tf.sparse_reorder doesn't requied since COO format is lexigraphical ordered

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
        fit_intercept=True,
        input_type='dense',
        log_dir=None, verbose=0
    ):

        self.rank = rank
        self.order = order
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.fit_intercept = fit_intercept
        self.reg = reg
        self.init_std = init_std
        self.input_type = input_type
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.verbose = verbose

        self.graph = None
        self.n_features = None
        self.steps = 0

    def initialize_graph(self):
        """Build computational graph according to params."""
        assert self.n_features is not None
        self.graph = tf.Graph()
        with self.graph.as_default():

            with tf.name_scope('params') as scope:
                self.w = [None] * self.order
                for i in range(1, self.order + 1):
                    r = self.rank
                    if i == 1:
                        r = 1
                    self.w[i - 1] = tf.Variable(
                        tf.random_uniform(
                            [self.n_features, r],
                            -self.init_std,
                            self.init_std),
                        trainable=True,
                        name='embedding_' + str(i))

                self.b = tf.Variable(
                    self.init_std,
                    trainable=True,
                    name='bias')
                tf.scalar_summary('bias', self.b)

            with tf.name_scope('inputBlock') as scope:
                if self.input_type == 'dense':
                    self.train_x = tf.placeholder(
                        tf.float32,
                        shape=[None, self.n_features],
                        name='x')
                else:
                    self.raw_indices = tf.placeholder(
                        tf.int64,
                        shape=[None, 2],
                        name='raw_indices')
                    self.raw_values = tf.placeholder(
                        tf.float32,
                        shape=[None],
                        name='raw_data')
                    self.raw_shape = tf.placeholder(
                        tf.int64, shape=[2],
                        name='raw_shape')
                    # tf.sparse_reorder is not needed since
                    # scipy return COO in canonical order
                    self.train_x = tf.SparseTensor(
                        self.raw_indices,
                        self.raw_values,
                        self.raw_shape)

                self.train_y = tf.placeholder(
                    tf.float32,
                    shape=[None],
                    name='Y')

            with tf.name_scope('mainBlock') as scope:
                self.outputs = self.b

                with tf.name_scope('linear_part') as scope:
                    if self.input_type == 'dense':
                        contribution = tf.matmul(self.train_x, self.w[0])
                    else:
                        contribution = tf.sparse_tensor_dense_matmul(
                            self.train_x,
                            self.w[0])
                self.outputs += contribution

                def pow_matmul(order, pow):
                    if pow not in pow_matmul.x_pow_cache:
                        pow_matmul.x_pow_cache[pow] = core.train_x_pow_wrapper(self, pow, self.input_type)
                    if order not in pow_matmul.matmul_cache:
                        pow_matmul.matmul_cache[order] = {}
                    if pow not in pow_matmul.matmul_cache[order]:
                        w_pow = tf.pow(self.w[order-1], pow)
                        pow_matmul.matmul_cache[order][pow] = core.matmul_wrapper(pow_matmul.x_pow_cache[pow], w_pow, self.input_type)
                    return pow_matmul.matmul_cache[order][pow]
                pow_matmul.x_pow_cache = {}
                pow_matmul.matmul_cache = {}

                for i in range(2, self.order + 1):
                    with tf.name_scope('order_{}'.format(i)) as scope:
                        # if self.input_type == 'dense':
                        raw_dot = core.matmul_wrapper(self.train_x, self.w[i - 1], self.input_type)
                        dot = tf.pow(raw_dot, i)
                        initialization_shape = tf.shape(dot)
                        for in_pows, out_pows, coef in utils.powers_and_coefs(i):
                            product_of_pows = tf.ones(initialization_shape)
                            for pow_idx in range(len(in_pows)):
                                product_of_pows *= tf.pow(pow_matmul(i, in_pows[pow_idx]), out_pows[pow_idx])
                            dot -= coef * product_of_pows
                        # else:
                        #     # TODO: Apply fixes from the dense code to the sparse code.
                        #     raw_dot = tf.sparse_tensor_dense_matmul(
                        #         self.train_x,
                        #         self.w[i - 1])
                        #     dot = tf.pow(raw_dot, i)
                        #     # tf.sparse_reorder is not needed
                        #     powered_x = tf.SparseTensor(
                        #         self.raw_indices,
                        #         tf.pow(self.raw_values, i),
                        #         self.raw_shape)
                        #     dot -= tf.sparse_tensor_dense_matmul(
                        #         powered_x,
                        #         tf.pow(self.w[i - 1], i))
                        contribution = tf.reshape(
                            tf.reduce_sum(dot, [1]),
                            [-1, 1])
                        contribution /= float(math.factorial(i))

                    self.outputs += contribution

                with tf.name_scope('loss') as scope:
                    self.probs = tf.sigmoid(self.outputs, name='probs')
                    self.loss = tf.minimum(
                        tf.log(
                            tf.add(
                                1.0,
                                tf.exp(
                                    -self.train_y * tf.transpose(
                                        self.outputs
                                    )
                                )
                            )
                        ),
                        100, name='truncated_log_loss')
                    self.reduced_loss = tf.reduce_mean(self.loss)
                    tf.scalar_summary('loss', self.reduced_loss)

                with tf.name_scope('regularization') as scope:
                    self.regularization = 0
                    for i in range(1, self.order + 1):
                        norm = tf.nn.l2_loss(
                            self.w[i - 1],
                            name='regularization_penalty_' + str(i))
                        tf.scalar_summary('norm_W_{}'.format(i), norm)
                        self.regularization += norm
                    tf.scalar_summary(
                        'regularization_penalty',
                        self.regularization)

            self.target = self.reduced_loss + self.reg * self.regularization
            self.checked_target = tf.verify_tensor_all_finite(
                self.target,
                msg='NaN or Inf in target value', name='target')
            tf.scalar_summary('target', self.checked_target)

            self.trainer = self.optimizer.minimize(self.checked_target)
            self.init = tf.initialize_all_variables()
            self.summary_op = tf.merge_all_summaries()
            self.saver = tf.train.Saver()

    def initialize_session(self):
        """Start computational session on builded graph.

        Initialize summary logger (if needed).
        """
        if self.graph is None:
            raise 'Graph not found. Try call initialize_graph() before initialize_session()'
        if self.need_logs:
            if os.path.exists(self.log_dir):
                print('log dir not empty -- delete it')
                shutil.rmtree(self.log_dir)
            self.summary_writer = tf.train.SummaryWriter(
                self.log_dir,
                self.graph)
            if self.verbose > 0:
                print('Initialize logs, use: \ntensorboard --logdir={}'.format(
                    os.path.abspath(self.log_dir)))
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

    def destroy(self):
        """Terminate session and destroyes graph."""
        self.session.close()
        self.graph = None

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

        for i in range(0, X_.shape[0], batch_size):
            ret_x = X_[i:i + batch_size]
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
        if self.input_type == 'dense':
            fd[self.train_x] = X.astype(np.float32)
        else:
            # sparse case
            X_sp = X.tocoo()
            fd[self.raw_indices] = np.hstack((
                X_sp.row[:, np.newaxis],
                X_sp.col[:, np.newaxis])).astype(np.int64)
            fd[self.raw_values] = X_sp.data.astype(np.float32)
            fd[self.raw_shape] = np.array(X_sp.shape).astype(np.int64)
        if y is not None:
            fd[self.train_y] = y.astype(np.float32)
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
        self.n_features = X_.shape[1]
        assert self.n_features is not None
        if self.graph is None:
            self.initialize_graph()
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
                    [self.trainer, self.target, self.summary_op],
                    feed_dict=fd)

                if self.verbose > 1:
                    w_sum = self.regularization.eval()
                    print ' -> batch: {}, target: {}, w_sum: {}'.format(
                        i,
                        batch_target_value,
                        w_sum)

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
        if self.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for (bX, bY) in self.batcher(X):
            fd = self.batch_to_feeddict(bX, bY)
            output.append(self.session.run(self.outputs, feed_dict=fd))
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
        if self.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for (bX, bY) in self.batcher(X):
            fd = self.batch_to_feeddict(bX, bY)
            output.append(self.session.run(self.probs, feed_dict=fd))
        probs_positive = np.concatenate(output)
        probs_negative = 1 - probs_positive
        probs = np.concatenate((probs_negative, probs_positive), axis=1)
        return probs

    def save_state(self, path):
        """Save current session to file.

        Parameters
        ----------
        path : str
            Destination file
        """
        self.saver.save(self.session, path)

    def load_state(self, path):
        """Restore session state from file.

        Parameters
        ----------
        path : str
            Restoring file
        """
        if self.graph is None:
            self.initialize_graph()
            self.initialize_session()
        self.saver.restore(self.session, path)
