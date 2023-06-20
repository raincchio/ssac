import numpy as np
import math

beta=100
number = 1000000
# x = np.random.randint(20, size=(1, 10))
#
# print(x)
# data = np.exp(x*beta - x.max()*beta) / np.exp(x*beta - x.max()*beta).sum()
# print(data)
# res = data*x
# print(res.sum(), x.max())


x = np.random.randint(1000, size=(1, number))
betax = x* beta

logsum = betax.max() + np.log(np.exp(betax-betax.max()).sum())

logsoftmax = betax - betax.max() - np.log(np.exp(betax-betax.max()).sum())

prob = np.exp(logsoftmax)

softmax = prob*x

entropy = 0.0

for value in prob.squeeze():
    if value>1e-5:
        entropy += -value*np.log(value)

print(logsum/beta, softmax.sum(), entropy/beta, x.max())