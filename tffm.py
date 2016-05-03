"""A tensorflow implementation of d-way FM (d>=2)
    Should support sklearn stuff like cross-validation.
"""
import sklearn
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.base import BaseEstimator


class TFFMClassifier(BaseEstimator):
    def __init__(self, rank, order=2, fit_intercept=True, lr=0.1, batch_size=-1, reg=0, init_std=0.01, n_epochs=100, verbose=0, nfeats=None):

        # Save all the params as class attributes. It's required by the
        # BaseEstimator class.
        self.rank = rank
        self.lr = lr
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.reg = reg
        self.order = order
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_std = init_std
        self.graph = None
        self.nfeats = None

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

            with tf.name_scope('inputBlock') as scope:
                self.train_x = tf.placeholder(tf.float32, shape=[None, self.nfeats], name='x')
                self.train_y = tf.placeholder(tf.float32, shape=[None], name='Y')

            with tf.name_scope('mainBlock') as scope:
                self.outputs = tf.matmul(self.train_x, self.w[0])
                for i in range(2, self.order+1):
                    dot = tf.pow(tf.matmul(self.train_x, self.w[i-1]), i)
                    # Subtract diagonal elements.
                    dot -= tf.matmul(tf.pow(self.train_x, i), tf.pow(self.w[i-1], i))
                    self.outputs += tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                self.outputs += self.b

                self.probs = tf.sigmoid(self.outputs, name='probs')
                self.loss = tf.minimum(
                    tf.log(tf.add(1.0, tf.exp(-self.train_y * tf.transpose(self.outputs)))), 
                    100, name='truncated_log_loss') 

                self.regularization = 0
                for i in range(1, self.order+1):
                    self.regularization += tf.nn.l2_loss(self.w[i-1], name='regularization_penalty_'+str(i))

            self.target = tf.reduce_mean(self.loss) + self.reg*self.regularization
            self.checked_target = tf.verify_tensor_all_finite(self.target, msg='NaN or Inf in target value', name='target_numeric_check')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.checked_target)

    def initialize_session(self):
        # TODO: get rid of interactive session.
        self.session = tf.InteractiveSession(graph=self.graph)
        init = tf.initialize_all_variables()
        self.session.run(init)
        # TODO: add SummaryWriter

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

    def fit(self, X_, y_, n_epochs=None, progress_bar=False):
        self.nfeats = X_.shape[1]
        assert self.nfeats is not None
        if self.graph is None:
            self.initialize_graph()
            self.initialize_session()
        used_y = y_*2 - 1 # suppose input {0, 1}, but use instead {-1, 1} labels

        if n_epochs is None:
            n_epochs = self.n_epochs

        # Training cycle
        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not progress_bar)):
            if self.verbose>0:
                print 'start epoch: {}'.format(epoch)

            # generate permutation
            perm = np.random.permutation(X_.shape[0])

            # iterate over batches
            for i, (bX, bY) in enumerate(self.batcher(X_[perm], used_y[perm])):
                fd = {
                    self.train_x : bX.astype(np.float32), 
                    self.train_y : bY.astype(np.float32),
                }
                self.session.run(self.optimizer, feed_dict=fd)

                if self.verbose > 1:
                    batch_target_value = self.session.run(self.target, feed_dict=fd)
                    w_sum = self.regularization.eval()
                    print ' -> batch: {}, target: {}, w_sum: {}'.format(i, batch_target_value, w_sum)

    def decision_function(self, X):
        if self.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        for (bX, bY) in self.batcher(X):
            fd = {
                self.train_x : bX.astype(np.float32), 
            }
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
            fd = {
                self.train_x : bX.astype(np.float32), 
            }
            output.append(self.session.run(self.probs, feed_dict=fd))
        probs_positive = np.concatenate(output)
        probs_negative = 1 - probs_positive
        return np.concatenate((probs_negative, probs_positive), axis=1)
