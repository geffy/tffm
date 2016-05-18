import tensorflow as tf


def matmul_wrapper(A, B, optype):
    """
    A, B: tf.Tensor
    optype: str, {'dense', 'sparse'}

    Return
    tf.Op
    """
    if optype == 'dense':
        return tf.matmul(A, B)
    elif optype == 'sparse':
        return tf.sparse_tensor_dense_matmul(A, B)
    else:
        raise NameError('Unknown input type in matmul')


def train_x_pow_wrapper(obj, p, optype):
    """
    obj: TFFMClassifier
    p: int
    optype: str, {'dense', 'sparse'}

    Return
    tf.Op
    """
    # TODO: assert existence of placeholders
    if optype == 'dense':
        return tf.pow(obj.train_x, p)
    elif optype == 'sparse':
        return tf.SparseTensor(
            obj.raw_indices,
            tf.pow(obj.raw_values, p),
            obj.raw_shape
        )
        print(type(obj.train_x))
    else:
        raise NameError('Unknown input type in matmul')
