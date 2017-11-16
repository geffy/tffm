import unittest
import numpy as np
from tffm import TFFMClassifier, TFFMRegressor
from scipy import sparse as sp
import tensorflow as tf
import pickle
import operator


class TestFM(unittest.TestCase):

    def setUp(self):
        # Reproducibility.
        np.random.seed(0)

        n_samples = 20
        n_features = 10

        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.binomial(1, 0.5, size=n_samples)
        self.s = np.random.uniform(low=.05, high=10, size=n_samples)
        self.s_ones = np.ones_like(self.y)

    def train_model(self, model_type='classifier', use_weights=False, s=None):
        if model_type == 'classifier':
            model = TFFMClassifier(
                order=1,
                rank=10,
                optimizer=tf.train.AdamOptimizer(learning_rate=1.0),
                n_epochs=2,
                init_std=1.0,
                seed=0,
                use_weights=use_weights
            )
        else:
            model = TFFMRegressor(
                order=1,
                rank=10,
                optimizer=tf.train.AdamOptimizer(learning_rate=1.0),
                n_epochs=2,
                init_std=1.0,
                seed=0,
                use_weights=use_weights
            )

        model.fit(self.X, self.y, s)
        w = model.weights
        model.destroy()
        return w

    def decision_function_order_4(self, input_type, use_diag=False):
        # Explanation for init_std=1.0.
        # With small init_std the contribution of higher order terms is
        # neglectable, so we would essentially test only low-order implementation.
        # That's why a relatively high init_std=1.0 here.
        model = TFFMClassifier(
            order=4,
            rank=10,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
            n_epochs=0,
            input_type=input_type,
            init_std=1.0,
            seed=0,
            use_diag=use_diag
        )

        if input_type == 'dense':
            X = self.X
        else:
            X = sp.csr_matrix(self.X)

        model.fit(X, self.y)
        b = model.intercept
        w = model.weights

        desired = self.bruteforce_inference(self.X, w, b, use_diag=use_diag)

        actual = model.decision_function(X)
        model.destroy()
        np.testing.assert_almost_equal(actual, desired, decimal=4)

    def test_dense_FM(self):
        self.decision_function_order_4(input_type='dense', use_diag=False)

    def test_dense_PN(self):
        self.decision_function_order_4(input_type='dense', use_diag=True)

    def test_sparse_FM(self):
        self.decision_function_order_4(input_type='sparse', use_diag=False)

    def test_sparse_PN(self):
        self.decision_function_order_4(input_type='sparse', use_diag=True)

    def test_classifier_using_weights_ones(self):
        actual = self.train_model(use_weights=True, s=self.s_ones)
        desired = self.train_model()
        np.testing.assert_almost_equal(actual, desired, decimal=4)
    
    def test_classifier_using_weights(self):
        w_trained_weights = self.train_model(use_weights=True, s=self.s)
        w_trained_no_weights = self.train_model()
        np.testing.utils.assert_array_compare(operator.__ne__, w_trained_no_weights, w_trained_weights)

    def test_regressor_using_weights_ones(self):
        actual = self.train_model(model_type='regressor', use_weights=True, s=self.s_ones)
        desired = self.train_model(model_type='regressor')
        np.testing.assert_almost_equal(actual, desired, decimal=4)
    
    def test_regressor_using_weights(self):
        w_trained_weights = self.train_model(model_type='regressor', use_weights=True, s=self.s)
        w_trained_no_weights = self.train_model(model_type='regressor')
        np.testing.utils.assert_array_compare(operator.__ne__, w_trained_no_weights, w_trained_weights)

    def bruteforce_inference_one_interaction(self, X, w, order, use_diag):
        n_obj, n_feat = X.shape
        ans = np.zeros(n_obj)
        if order == 2:
            for i in range(n_feat):
                for j in range(0 if use_diag else i+1, n_feat):
                    x_prod = X[:, i] * X[:, j]
                    w_prod = np.sum(w[1][i, :] * w[1][j, :])
                    denominator = 2.0**(order-1) if use_diag else 1.0
                    ans += x_prod * w_prod / denominator
        elif order == 3:
            for i in range(n_feat):
                for j in range(0 if use_diag else i+1, n_feat):
                    for k in range(0 if use_diag else j+1, n_feat):
                        x_prod = X[:, i] * X[:, j] * X[:, k]
                        w_prod = np.sum(w[2][i, :] * w[2][j, :] * w[2][k, :])
                        denominator = 2.0**(order-1) if use_diag else 1.0
                        ans += x_prod * w_prod / denominator
        elif order == 4:
            for i in range(n_feat):
                for j in range(0 if use_diag else i+1, n_feat):
                    for k in range(0 if use_diag else j+1, n_feat):
                        for ell in range(0 if use_diag else k+1, n_feat):
                            x_prod = X[:, i] * X[:, j] * X[:, k] * X[:, ell]
                            w_prod = np.sum(w[3][i, :] * w[3][j, :] * w[3][k, :] * w[3][ell, :])
                            denominator = 2.0**(order-1) if use_diag else 1.0
                            ans += x_prod * w_prod / denominator
        else:
            assert False
        return ans

    def bruteforce_inference(self, X, w, b, use_diag):
        assert len(w) <= 4
        ans = X.dot(w[0]).flatten() + b
        if len(w) > 1:
            ans += self.bruteforce_inference_one_interaction(X, w, 2, use_diag)
        if len(w) > 2:
            ans += self.bruteforce_inference_one_interaction(X, w, 3, use_diag)
        if len(w) > 3:
            ans += self.bruteforce_inference_one_interaction(X, w, 4, use_diag)
        return ans


if __name__ == '__main__':
    unittest.main()
