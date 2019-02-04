This is a TensorFlow implementation of an arbitrary order (>=2) Factorization Machine based on paper [Factorization Machines with libFM](http://dl.acm.org/citation.cfm?doid=2168752.2168771).

It supports:
* dense and sparse inputs
* different (gradient-based) optimization methods
* classification/regression via different loss functions (logistic and mse implemented)
* logging via TensorBoard

The inference time is linear with respect to the number of features.

Tested on Python3.5, but should work on Python2.7

This implementation is quite similar to the one described in Blondel's et al. paper [https://arxiv.org/abs/1607.07195], but was developed independently and prior to the first appearance of the paper.

# Dependencies
* [scikit-learn](http://scikit-learn.org/stable/)
* [numpy](http://www.numpy.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [tensorflow 1.0+ (tested on 1.3)](https://www.tensorflow.org/)

# Installation
Stable version can be installed via `pip install tffm`. 

# Usage
The interface is similar to scikit-learn models. To train a 6-order FM model with rank=10 for 100 iterations with learning_rate=0.01 use the following sample
```python
from tffm import TFFMClassifier
model = TFFMClassifier(
    order=6,
    rank=10,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=100,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)
model.fit(X_tr, y_tr, show_progress=True)
```

See `example.ipynb` and `gpu_benchmark.ipynb` for more details.

It's highly recommended to read `tffm/core.py` for help.


# Testing
Just run ```python test.py``` in the terminal. ```nosetests``` works too, but you must pass the `--logging-level=WARNING` flag to avoid printing insane amounts of TensorFlow logs to the screen.


# Citation
If you use this software in academic research, please, cite it using the following BibTeX:
```latex
@misc{trofimov2016,
author = {Mikhail Trofimov, Alexander Novikov},
title = {tffm: TensorFlow implementation of an arbitrary order Factorization Machine},
year = {2016},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/geffy/tffm}},
}
```
