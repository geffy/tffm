import tensorflow as tf


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
        raise NameError('Unknown input type in pow_wrapper')
