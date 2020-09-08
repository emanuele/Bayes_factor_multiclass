#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

"""Implementation of the algorithms for testing independence
in contingency tables as proposed in:

Assessing Robustness of Intrinsic Tests of Independence in Two-Way
Contingency Tables, George Casella and Elías Moreno, Journal of the
American Statistical Association, Vol. 104, No. 487. (September 2009),
pp. 1261-1271. doi:10.1198/jasa.2009.tm08106

WARNING: this implementation slightly differs from the equations in
that paper because of some typographical errors appearing there.

Copyright Emanuele Olivetti, 2011.

This program is distributed under the GNU General Public Licence v3.0.
"""

import numpy as np
from scipy.special import factorial
from scipy.special import gammaln


def multinomial_coefficient(n, x):
    """Multinomial coefficient. See:
    http://en.wikipedia.org/wiki/Binomial_coefficient#Generalization_to_multinomials
    """
    return factorial(n) / factorial(x).prod()


def log_multinomial_coefficient(n, x):
    """Log of the multinomial coefficient computed through gammaln().
    """
    return gammaln(n + 1) - gammaln(x + 1).sum()

def B_10_basic(y, t, M=30000, verbose=True):
    """MonteCarlo estimate of the Bayes factor
    BF_10 = p(y|H_dependent) / p(y|H_independent)
    using the mutlinomial distribution as candidate distribution
    for the importance sampling strategy.

    This is the straightforward/naive implementation from the
    equations in the paper. This implementation is both numerically
    unstable and slow.
    
    y = contingency matrix
    t = concentration parameter, sample size
    M = iterations of the MC approximation
    """
    n = y.sum()
    a, b = y.shape
    theta_hat = (y + 1.0) / (n + a * b)
    B_10 = factorial(t + a * b - 1) / factorial(t + n + a * b - 1)
    B_10 *= factorial(n + a - 1) * factorial(n + b - 1)
    B_10 /= factorial(t + a - 1) * factorial(t + b - 1)
    tmp_sum = 0.0
    for k in range(M):
        x_k = np.random.multinomial(n=t, pvals=theta_hat.flatten()).reshape(a, b)
        tmp = 1.0
        tmp *= multinomial_coefficient(t, x_k)
        tmp *= factorial(x_k.sum(1)).prod()
        tmp /= factorial(y.sum(1)).prod()
        tmp *= factorial(x_k.sum(0)).prod()
        tmp /= factorial(y.sum(0)).prod()
        tmp *= factorial(x_k + y).prod()
        tmp /= factorial(x_k).prod()
        tmp /= multinomial_coefficient(t, x_k)
        tmp /= (theta_hat**x_k).prod()
        tmp_sum += tmp
    B_10 *= tmp_sum  / np.double(M)
    return B_10

        
def log_B_10(y, t, M=30000, verbose=False):
    """MonteCarlo estimate of the log of the Bayes factor
    BF_10 = p(y|H_dependent) / p(y|H_independent)
    using the mutlinomial distribution as candidate distribution
    for the importance sampling strategy.

    This implementation is numerically more stable than B_10() due to
    the logscale, the gamma function instead of factorial and the use
    of np.logaddexp() function. It is faster too (6x).
    
    y = contingency matrix
    t = concentration parameter, sample size
    M = iterations of the MC approximation
    """
    n = y.sum()
    a, b = y.shape
    theta_hat = (y + 1.0) / (n + a * b)
    log_B_10 = gammaln(t + a * b) - gammaln(t + n + a * b)
    log_B_10 += gammaln(n + a) + gammaln(n + b) - gammaln(t + a) - gammaln(t + b)
    log_tmp = np.zeros(M)
    Ry = gammaln(y.sum(1) + 1).sum()
    Cy = gammaln(y.sum(0) + 1).sum()
    for k in range(M):
        x_k = np.random.multinomial(n=t, pvals=theta_hat.flatten()).reshape(a, b)
        log_tmp_i = gammaln(x_k.sum(1) + 1).sum() - Ry
        log_tmp_j = gammaln(x_k.sum(0) + 1).sum() - Cy
        log_tmp_ij = gammaln(x_k + y + 1).sum()
        log_tmp_ij -= gammaln(x_k + 1).sum()
        log_tmp_ij -= (x_k*np.log(theta_hat)).sum()
        log_tmp[k] = log_tmp_i + log_tmp_j + log_tmp_ij
        if verbose:
            print log_tmp[k] , log_tmp_i , log_tmp_j , log_tmp_ij
    log_B_10 += np.logaddexp.reduce(log_tmp) - np.log(M)
    return log_B_10

def log_B_10_fast(y, t, M=30000, verbose=False):
    """MonteCarlo estimatre of the log of the Bayes factor
    BF_10 = p(y|H_dependent) / p(y|H_independent)
    using the mutlinomial distribution as candidate distribution
    for the importance sampling strategy.
    
    Fast implementation (100x) that exploits broadcasting and removes
    the foor loop.

    y = contingency matrix
    t = concentration parameter, sample size
    M = iterations of the MC approximation
    """
    n = y.sum()
    a, b = y.shape
    theta_hat = (y + 1.0) / (n + a * b)
    log_B_10 = gammaln(t + a * b) - gammaln(t + n + a * b)
    log_B_10 += gammaln(n + a) + gammaln(n + b) - gammaln(t + a) - gammaln(t + b)
    log_tmp = np.zeros(M)
    Ry = gammaln(y.sum(1) + 1).sum()
    Cy = gammaln(y.sum(0) + 1).sum()
    x = np.random.multinomial(n=t, pvals=theta_hat.flatten(), size=M).reshape(M, a, b)
    log_tmp_i = gammaln(x.sum(2) + 1).sum(-1) - Ry
    log_tmp_j = gammaln(x.sum(1) + 1).sum(-1) - Cy
    log_tmp_ij = gammaln(x + y + 1).sum(-1).sum(-1)
    log_tmp_ij -= gammaln(x + 1).sum(-1).sum(-1)
    log_tmp_ij -= (x * np.log(theta_hat)).sum(-1).sum(-1)
    log_tmp = log_tmp_i + log_tmp_j + log_tmp_ij
    log_B_10 += np.logaddexp.reduce(log_tmp) - np.log(M)
    return log_B_10


if __name__=='__main__':

    np.random.seed(0)

    M = 100000

    expected_availbale = False

    table_11 = {'y': np.array([[2, 2, 2],
                               [2, 2, 2],
                               [2, 2, 2]]),
                't=1': 0.648,
                't=n': 0.701}
    
    table_12 = {'y': np.array([[6, 6, 6],
                               [6, 6, 6],
                               [6, 6, 6]]),
                't=1': 0.891,
                't=n': 0.839}
    
    table_15 = {'y': np.array([[5, 0, 0],
                               [5, 0, 0],
                               [5, 0, 0]]),
                't=1': 0.964,
                't=n': 0.872}

    table_7_mendel = {'y': np.array([[30,  60, 28],
                                     [65, 138, 68],
                                     [35,  67, 30]]),
                      't=1':  0.997,
                      't=n':  0.823}

    table_6_fienberg = {'y': np.array([[225, 53, 206],
                                       [  3,  1,  12]]),
                        't=1':  0.933,
                        't=n':  0.241}

    table = table_11
    y = table['y']
    t = 1
    print "(y_ij):"
    print y
    print "t =", t
    print "M =", M
    print

    if t==1 or t==y.sum():
        expected_availbale = True
        p_M0_given_y_t_expected = table['t='+str(t if t!=y.sum() else 'n')]
    else:
        print "The expected values for the chosen t are not available."

    if expected_availbale:
        B_10_expected = 1.0 / p_M0_given_y_t_expected - 1.0
        print "B_10 (expected) =", B_10_expected
    B_10_basic = B_10_basic(y, t=t, M=M)
    print "B_10 (basic)    =", B_10_basic
    log_B_10 = log_B_10(y, t=t, M=M)
    B_10 = np.exp(log_B_10)
    print "B_10 (gammaln)  =", B_10
    log_B_10_fast = log_B_10_fast(y, t=t, M=M)
    B_10_fast = np.exp(log_B_10_fast)
    print "B_10 (fast)     =", B_10_fast
    
    print
    if expected_availbale:
        print "p(M0|y,t) (expected) =", p_M0_given_y_t_expected
    print "p(M0|y,t) (basic)    =", 1.0 / (1.0 + B_10_basic)
    print "p(M0|y,t) (gammaln)  =", 1.0 / (1.0 + B_10)
    print "p(M0|y,t) (fast)     =", 1.0 / (1.0 + B_10_fast)
