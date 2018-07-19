# SimpleLDA
A C++ implemented LDA (Latent Dirichlet Allocation) with a python wrapper

This work is inspired by https://github.com/lda-project/lda which use python + Cython, and some of the code is in pure python which will be slow. This project write all LDA in C++ and is faster than the origional one.

## Usage

### To complie the code:
```
   git clone https://github.com/fuzihaofzh/SimpleLDA.git
   cd SimpleLDA
   make
```

### Use with python:
```
import numpy as np
import ldalib
X = np.random.randint(0, 5, (5, 3))
doc_topic, topic_word = ldalib.lda(X, 10, iter_n = 100)
```
