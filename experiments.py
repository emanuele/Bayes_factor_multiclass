import numpy as np
from partial_independence import compute_logp_H
from partitioner import Partition
from icann2011_confusion_matrices import *

if __name__ == '__main__':

    X = np.array([[10,10, 0],
                  [10,10, 0],
                  [ 0, 0,20]])

    X = np.array([[13, 3, 4],
                  [ 3,14, 3],
                  [ 4, 3,13]])

    # X = np.array([[10, 10, 10,  0,  0],
    #               [10, 10, 10,  0,  0],
    #               [10, 10, 10,  0,  0],
    #               [ 0,  0,  0, 30,  0],
    #               [ 0,  0,  0,  0, 30]], dtype=np.float32)

    X = X

    # X = huttunen  # tu  # jylanki # santana
    X = tu
    
    print "X:"
    print X

    psi = [[0,1],[2]]
    # psi = [[0],[1],[2]]
    # psi = [[0],[1,2]]
    # psi = [[0,1,2]]
    # psi = [[0],[1],[2],[3],[4]]
    # psi = [[0,1],[2],[3],[4]]
    # psi = [[0,1,2],[3],[4]]
    # psi = [[0,1,2,3],[4]]
    # psi = [[0,1,2],[3,4]]
    # psi = [[0,1,2,3,4]]

    print "psi:", psi

    alpha = np.ones(X.shape)
    print "alpha:"
    print alpha

    logp_Hs = []
    print "psi \t log-likelihood"
    partitions = Partition(range(X.shape[0]))
    partitions1 = Partition(range(1,X.shape[0]+1))
    for psi in partitions:
        logp_H = compute_logp_H(X, alpha, psi)
        print psi, logp_H
        logp_Hs.append(logp_H)

    idxs = np.argsort(logp_Hs)
    print
    for idx in idxs:
        print partitions[idx], logp_Hs[idx]
    
