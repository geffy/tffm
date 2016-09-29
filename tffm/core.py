import tensorflow as tf
from . import utils
import math


class TFFMCore():
    """
    This class underlying routine about creating computational graph.

    Its required n_features to be set at graph building time.


    Parameters
    ----------

    order : int, default: 2
        Order of corresponding polynomial model.
        All interaction from bias and linear to order will be included.

    rank : int
        Number of factors in low-rank appoximation.
        This value is shared across different orders of interaction.

    input_type : str, 'dense' or 'sparse', default: 'dense'
        Type of input data. Only numpy.array allowed for 'dense' and
        scipy.sparse.csr_matrix for 'sparse'. This affects construction of
        computational graph and cannot be changed during training/testing.

    optimizer : tf.train.Optimizer, default: AdamOptimizer(learning_rate=0.1)
        Optimization method used for training

    reg : float, default: 0
        Strength of L2 regularization

    init_std : float, default: 0.01
        Amplitude of random initialization


    Attributes
    ----------
    graph : tf.Graph or None
        Initialized computational graph or None

    trainer : tf.Op
        TensorFlow operation node to perform learning on single batch

    n_features : int
        Number of features used in this dataset.
        Inferred during the first call of fit() method.

    saver : tf.Op
        tf.train.Saver instance, connected to graph

    summary_op : tf.Op
        tf.merge_all_summaries instance for export logging

    b : tf.Variable, shape: [1]
        Bias term.

    w : array of tf.Variable, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    Notes
    -----
    Parameter rank is shared across all orders of interactions (except bias and
    linear parts).
    tf.sparse_reorder doesn't requied since COO format is lexigraphical ordered.
    This implementation uses a generalized approach from referenced paper along
    with caching.

    References
    ----------
    Steffen Rendle, Factorization Machines
        http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """
    def __init__(self, order, rank, input_type, loss_function, optimizer, reg, init_std, seed=None):
        self.order = order
        self.rank = rank
        self.input_type = input_type
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.reg = reg
        self.init_std = init_std
        self.seed = seed
        self.n_features = None
        self.graph = None

    def set_num_features(self, n_features):
        self.n_features = n_features

    def init_learnable_params(self):
        self.w = [None] * self.order
        for i in range(1, self.order + 1):
            r = self.rank
            if i == 1:
                r = 1
            rnd_weights = tf.random_uniform([self.n_features, r], -self.init_std, self.init_std)
            self.w[i - 1] = tf.Variable(rnd_weights, trainable=True, name='embedding_' + str(i))
        self.b = tf.Variable(self.init_std, trainable=True, name='bias')
        tf.scalar_summary('bias', self.b)

    def init_placeholders(self):
        if self.input_type == 'dense':
            self.train_x = tf.placeholder(tf.float32, shape=[None, self.n_features], name='x')
        else:
            self.raw_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
            self.raw_values = tf.placeholder(tf.float32, shape=[None], name='raw_data')
            self.raw_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
            # tf.sparse_reorder is not needed since scipy return COO in canonical order
            self.train_x = tf.SparseTensor(self.raw_indices, self.raw_values, self.raw_shape)

        self.train_y = tf.placeholder(tf.float32, shape=[None], name='Y')

    def pow_matmul(self, order, pow):
        if pow not in self.x_pow_cache:
            x_pow = pow_wrapper(self.train_x, pow, self.input_type)
            self.x_pow_cache[pow] = x_pow
        if order not in self.matmul_cache:
            self.matmul_cache[order] = {}
        if pow not in self.matmul_cache[order]:
            w_pow = tf.pow(self.w[order - 1], pow)
            dot = matmul_wrapper(self.x_pow_cache[pow], w_pow, self.input_type)
            self.matmul_cache[order][pow] = dot
        return self.matmul_cache[order][pow]

    def init_loss(self):
        self.loss = self.loss_function(self.outputs, self.train_y)
        self.reduced_loss = tf.reduce_mean(self.loss)
        tf.scalar_summary('loss', self.reduced_loss)

    def init_regularization(self):
        self.regularization = 0
        for i in range(1, self.order + 1):
            node_name = 'regularization_penalty_' + str(i)
            norm = tf.nn.l2_loss(self.w[i - 1], name=node_name)
            tf.scalar_summary('norm_W_{}'.format(i), norm)
            self.regularization += norm
        tf.scalar_summary('regularization_penalty', self.regularization)

    def init_main_block(self):
        self.x_pow_cache = {}
        self.matmul_cache = {}

        self.outputs = self.b

        with tf.name_scope('linear_part') as scope:
            contribution = matmul_wrapper(self.train_x, self.w[0], self.input_type)
        self.outputs += contribution

        for i in range(2, self.order + 1):
            with tf.name_scope('order_{}'.format(i)) as scope:
                raw_dot = matmul_wrapper(self.train_x, self.w[i - 1], self.input_type)
                dot = tf.pow(raw_dot, i)
                initialization_shape = tf.shape(dot)
                for in_pows, out_pows, coef in utils.powers_and_coefs(i):
                    product_of_pows = tf.ones(initialization_shape)
                    for pow_idx in range(len(in_pows)):
                        product_of_pows *= tf.pow(
                            self.pow_matmul(i, in_pows[pow_idx]),
                            out_pows[pow_idx]
                        )
                    dot -= coef * product_of_pows
                contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                contribution /= float(math.factorial(i))
            self.outputs += contribution

        with tf.name_scope('loss') as scope:
            self.init_loss()

        with tf.name_scope('regularization') as scope:
            self.init_regularization()

    def init_target(self):
        self.target = self.reduced_loss + self.reg * self.regularization
        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value', name='target')
        tf.scalar_summary('target', self.checked_target)

    def build_graph(self):
        """Build computational graph according to params."""
        assert self.n_features is not None
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            with tf.name_scope('params') as scope:
                self.init_learnable_params()

            with tf.name_scope('inputBlock') as scope:
                self.init_placeholders()

            with tf.name_scope('mainBlock') as scope:
                self.init_main_block()

            self.init_target()

            self.trainer = self.optimizer.minimize(self.checked_target)
            self.init_all_vars = tf.initialize_all_variables()
            self.summary_op = tf.merge_all_summaries()
            self.saver = tf.train.Saver()


def matmul_wrapper(A, B, optype):
    """Wrapper for handling sparse and dense versions of matmul operation.

    Parameters
    ----------
    A : tf.Tensor
    B : tf.Tensor
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor
    """
    if optype == 'dense':
        return tf.matmul(A, B)
    elif optype == 'sparse':
        return tf.sparse_tensor_dense_matmul(A, B)
    else:
        raise NameError('Unknown input type in matmul_wrapper')


def pow_wrapper(X, p, optype):
    """Wrapper for handling sparse and dense versions of power operation.

    Parameters
    ----------
    X : tf.Tensor
    p : int
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor
    """
    if optype == 'dense':
        return tf.pow(X, p)
    elif optype == 'sparse':
        return tf.SparseTensor(X.indices, tf.pow(X.values, p), X.shape)
    else:
        raise NameError('Unknown input type in pow_wrapper')
