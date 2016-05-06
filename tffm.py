"""A tensorflow implementation of d-way FM (d>=2)
    Should support sklearn stuff like cross-validation.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sklearn
from sklearn.base import BaseEstimator
import time
import os


class TFFMClassifier(BaseEstimator):
    def __init__(self, rank, order=2, 
                optimizer=tf.train.AdamOptimizer(learning_rate=0.1), batch_size=-1, n_epochs=100,
                reg=0, init_std=0.01, fit_intercept=True, verbose=0, input_type='dense', log_dir=None):

        self.rank = rank
        self.order = order
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.fit_intercept = fit_intercept
        self.reg = reg
        self.init_std = init_std
        self.verbose = verbose
        self.input_type = input_type
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        
        self.graph = None
        self.nfeats = None
        self.steps = 0

    def initialize_graph(self):
        assert self.nfeats is not  None
        self.graph = tf.Graph()
        with self.graph.as_default():

            with tf.name_scope('params') as scope: 
                self.w = [None] * self.order
                for i in range(1, self.order+1):
                    r = self.rank
                    if i == 1:
                        r = 1
                    self.w[i-1] = tf.Variable(tf.random_uniform([self.nfeats, r], -self.init_std, self.init_std),
                                              trainable=True,
                                              name='embedding_'+str(i))

                self.b = tf.Variable(
                    self.init_std,
                    trainable = True,
                    name='bias')
                tf.scalar_summary('bias', self.b)

            with tf.name_scope('inputBlock') as scope:
                if self.input_type=='dense':
                    self.train_x = tf.placeholder(tf.float32, shape=[None, self.nfeats], name='x')
                else:
                    self.raw_indices = tf.placeholder(tf.int64, shape=[None, 2], name='raw_indices')
                    self.raw_values = tf.placeholder(tf.float32, shape=[None], name='raw_data')
                    self.raw_shape = tf.placeholder(tf.int64, shape=[2], name='raw_shape')
                    # tf.sparse_reorder is not needed -- scipy return COO in canonical order
                    self.train_x = tf.SparseTensor(
                                            self.raw_indices,
                                            self.raw_values,
                                            self.raw_shape
                                        )

                self.train_y = tf.placeholder(tf.float32, shape=[None], name='Y')

            with tf.name_scope('mainBlock') as scope:
                if self.input_type=='dense':
                    self.outputs = tf.matmul(self.train_x, self.w[0])
                else:
                    self.outputs = tf.sparse_tensor_dense_matmul(self.train_x, self.w[0])

                for i in range(2, self.order+1):
                    if self.input_type=='dense':
                        raw_dot = tf.matmul(self.train_x, self.w[i-1])
                        dot = tf.pow(raw_dot, i)
                        dot -= tf.matmul(tf.pow(self.train_x, i), tf.pow(self.w[i-1], i))
                    else:
                        raw_dot = tf.sparse_tensor_dense_matmul(self.train_x, self.w[i-1])
                        dot = tf.pow(raw_dot, i)
                        # tf.sparse_reorder is not needed -- scipy return COO in canonical order
                        powered_x = tf.SparseTensor(
                                        self.raw_indices,
                                        tf.pow(self.raw_values, i),
                                        self.raw_shape
                                    )
                        dot -= tf.sparse_tensor_dense_matmul(powered_x, tf.pow(self.w[i-1], i))

                    self.outputs += tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                self.outputs += self.b

                self.probs = tf.sigmoid(self.outputs, name='probs')
                self.loss = tf.minimum(
                    tf.log(tf.add(1.0, tf.exp(-self.train_y * tf.transpose(self.outputs)))), 
                    100, name='truncated_log_loss') 

                self.regularization = 0
                for i in range(1, self.order+1):
                    norm = tf.nn.l2_loss(self.w[i-1], name='regularization_penalty_'+str(i))
                    tf.scalar_summary('norm_W_{}'.format(i), norm)
                    self.regularization += norm

            self.reduced_loss = tf.reduce_mean(self.loss) 
            tf.scalar_summary('loss', self.reduced_loss)
            tf.scalar_summary('regularization_penalty', self.regularization)

            self.target = self.reduced_loss + self.reg*self.regularization
            self.checked_target = tf.verify_tensor_all_finite(self.target, msg='NaN or Inf in target value', name='target_numeric_check')
            tf.scalar_summary('target', self.checked_target)

            self.trainer = self.optimizer.minimize(self.checked_target)
            self.init = tf.initialize_all_variables()
            self.summary_op = tf.merge_all_summaries()

    def initialize_session(self):
        if self.graph is None:
            raise 'Graph not found. Try call initialize_graph() before initialize_session()'
        if self.need_logs:
            self.summary_writer = tf.train.SummaryWriter(self.log_dir, self.graph)
            if self.verbose>0:
                print('Initialize logs, use: \ntensorboard --logdir={}'.format(os.path.abspath(self.log_dir)))
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

    def destroy(self):
        self.session.close()
        self.graph = None

    def batcher(self, X_, y_=None):
        if self.batch_size == -1:
            batch_size = X_.shape[0]
        else:
            batch_size = self.batch_size

        for i in range(0, X_.shape[0], batch_size):
            retX = X_[i:i+batch_size]
            retY = None
            if y_ is not None:
                retY = y_[i:i+batch_size]
            yield (retX, retY)

    def batch_to_feeddict(self, X, y):
        fd = {}
        if self.input_type == 'dense':
            fd[self.train_x] = X.astype(np.float32)
        else:
            X_sp = X.tocoo()
            shape = X_sp.shape
            fd[self.raw_indices] = np.hstack((
                                        X_sp.row[:, np.newaxis],
                                        X_sp.col[:, np.newaxis]
                                    )).astype(np.int64)
            fd[self.raw_values] = X_sp.data.astype(np.float32)
            fd[self.raw_shape] = np.array(X_sp.shape).astype(np.int64)
        if y is not None:
            fd[self.train_y] = y.astype(np.float32)
        return fd

    def fit(self, X_, y_, n_epochs=None, show_progress=False):
        self.nfeats = X_.shape[1]
        assert self.nfeats is not None
        if self.graph is None:
            self.initialize_graph()
            self.initialize_session()
        used_y = y_*2 - 1 # suppose input {0, 1}, but use instead {-1, 1} labels

        if n_epochs is None:
            n_epochs = self.n_epochs

        # Training cycle
        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):
            if self.verbose>1:
                print 'start epoch: {}'.format(epoch)

            # generate permutation
            perm = np.random.permutation(X_.shape[0])

            # iterate over batches
            for i, (bX, bY) in enumerate(self.batcher(X_[perm], used_y[perm])):
                fd = self.batch_to_feeddict(bX, bY)
                _, batch_target_value, summary_str = self.session.run([self.trainer, self.target, self.summary_op], feed_dict=fd)

                if self.verbose > 1:
                    w_sum = self.regularization.eval()
                    print ' -> batch: {}, target: {}, w_sum: {}'.format(i, batch_target_value, w_sum)

                # Write stats
                if self.need_logs:
                    self.summary_writer.add_summary(summary_str, self.steps)
                    self.summary_writer.flush()
                self.steps += 1


    def decision_function(self, X):
        if self.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for (bX, bY) in self.batcher(X):
            fd = self.batch_to_feeddict(bX, bY)
            output.append(self.session.run(self.outputs, feed_dict=fd))
        return np.concatenate(output).reshape(-1)

    def predict(self, X):
        raw_output = self.decision_function(X)
        return (raw_output > 0).astype(int)

    def predict_proba(self, X):
        if self.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for (bX, bY) in self.batcher(X):
            fd = self.batch_to_feeddict(bX, bY)
            output.append(self.session.run(self.probs, feed_dict=fd))
        probs_positive = np.concatenate(output)
        probs_negative = 1 - probs_positive
        return np.concatenate((probs_negative, probs_positive), axis=1)
