# Intentionality-related Deep Learning method in Web Prefetching

This study develops an intentionality-related long short-term memory (Ir-LSTM) model as a prediction method.

# Requirements

All functions run in a Python3 environment.
Libraries: Gensim, numpy, Pytorch, matplotlib, BloomFilter, CountMinSketch

* Tips: If import errors happen, just use the pip3 to install the corresponding lib*

#### Files

`RNN_pytorch` : Key funcitons are provided here:

- Run **visiual()** to show the heat map of 2 variables
- Run **simulation()** to show the loading condition in a stream simulation.
- Run **testing _3D_model()**  to test all the possible results in different threshold and num of targets then save the result in a 3d matrix file
- Run **testing()**  to test all the possible results in different threshold and num of targets
- Run **training()** to train the Ir-LSTM model.
- Run **streaming_testing()** to test the models performance in a stream environment.