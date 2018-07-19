import ctypes
import numpy as np
def lda(X, K = 10, iter_n = 20, alpha = 0.1, beta = 0.01):
    X = X.astype("uint64")
    so = ctypes.CDLL("./llda.so")
    M, N = X.shape
    topic_word = np.empty(shape = (K, N), dtype = 'float32')
    doc_topic = np.empty(shape = (M, K), dtype = 'float32')
    so.lda(ctypes.c_void_p(X.ctypes.data), ctypes.c_uint64(M), ctypes.c_uint64(N), ctypes.c_uint64(K), ctypes.c_void_p(doc_topic.ctypes.data), ctypes.c_void_p(topic_word.ctypes.data), ctypes.c_uint64(iter_n), ctypes.c_float(alpha), ctypes.c_float(beta))
    return doc_topic, topic_word

if __name__ == '__main__':
    M = 2
    N = 3
    #X = np.random.randint(0, 5, (M, N))
    X = np.array([[1,0,3], [4,2,0]])
    doc_topic, topic_word = lda(X, 2)
