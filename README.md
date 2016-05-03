This is a TensorFlow implementation of Factorization Machine based on paper [Factorization Machines with libFM](http://dl.acm.org/citation.cfm?doid=2168752.2168771). It supports `d`-order FM, where `d>=2`


# Prerequisites
* [Scikit-learn](http://scikit-learn.org/stable/)
* [Numpy](http://www.numpy.org/)
* [TensorFlow 0.8](https://www.tensorflow.org/)

# Usage
The interface is the same as of Scikit-learn models. To train a 6-order FM model with rank=10 for 100 iterations with learning_rate=0.01 use the following sample
```
model = tffm.TFFMClassifier(order=6, rank=10, n_epochs=100,  lr=0.01)
model.fit(X_train, y_train)
```
