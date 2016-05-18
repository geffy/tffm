import tensorflow as tf


def matmul_wrapper(A, B, optype):
    """
    A, B: tf.Tensor
    optype: str, {'dense', 'sparse'}

    Return
    tf.Tensor
    """
    if optype == 'dense':
        return tf.matmul(A, B)
    elif optype == 'sparse':
        return tf.sparse_tensor_dense_matmul(A, B)
    else:
        raise NameError('Unknown input type in matmul')


def pow_wrapper(X, p, optype):
    """
    X: tf.SparseTensor
    p: int
    optype: str, {'dense', 'sparse'}

    Return
    tf.Tensor
    """
    # TODO: assert existence of placeholders
    if optype == 'dense':
        return tf.pow(X, p)
    elif optype == 'sparse':
        return tf.SparseTensor(
            X.indices,
            tf.pow(X.values, p),
            X.shape
        )
    else:
        raise NameError('Unknown input type in matmul')
