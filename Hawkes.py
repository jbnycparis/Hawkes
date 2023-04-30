from __future__ import print_function
import csv
import math
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize

DisplayPlots = True

# parameters ------------------
lambda0 = 1.2
alpha = 0.6
beta = 0.8
T = 1000
# simulation of the Hawkes process ------------------------
mu = lambda0
t = 0
ts = [0]
lambdas = [lambda0]
eps = 1e-6
while t < T:
    M = mu + (lambdas[-1] - mu) * np.exp(-beta * (t - ts[-1] + eps))
    u = np.random.uniform(0,1)
    t = t - np.log(u) / M
    v = np.random.uniform(0,1)
    if ( v <= (mu + (lambdas[-1] - mu) * np.exp(-beta * (t - ts[-1]))) / M ):
        ts.append(t)
        lambdas.append((lambdas[-1] - mu) * np.exp(-beta * (ts[-1] - ts[-2])) + mu + alpha)
ts = ts[1:-1]
lambdas = lambdas[1:-1]
n_events = len(ts)
print("n events=",n_events)
#print("ts=",ts)
dt = 0.01
nt = int(T/dt)
lambda_t = lambda0 * np.ones(nt)
events_t = np.zeros(nt)
time = (T/nt) * np.array(range(nt))
ind = 0
indices = []
for k in range(n_events):
    new_ind = int((ts[k] / T) * (nt - 1))
    if new_ind > ind:
        ind = new_ind
        indices.append(ind)
        events_t[ind] = 1
        lambda_t[ind:] += alpha * np.exp(-beta * (time[ind:] - ts[k]))

# calibration --------------
def NegLogLikelihood(X):
    lam_c = X[0]
    alpha_c = X[1]
    beta_c = X[2]
    # computation of R:
    R = [np.exp(-beta_c * ts[0])]
    for i in range(1,n_events):
        R.append(np.exp(-beta_c * ( ts[i] - ts[i-1] )) * (1. + R[-1]))
    # compute the log-likelihood
    ll = ts[-1]
    ll -= ts[0] * lam_c
    for i in range(1,n_events):
        ll -= lam_c * ( ts[i] - ts[i-1] )
        ll -= (alpha_c/beta_c) * (1. - np.exp(-beta_c * (ts[-1] - ts[i])))
        ll += np.log( lam_c + alpha_c * R[i] )
    return -ll
lamb_c0 = 0.5
alpha_c0 = 0.5
beta_c0 = 0.5
X0 = [lamb_c0,alpha_c0,beta_c0]
bnds = ((0.01, None), (0.01, None), (0.01,None))
res = minimize(NegLogLikelihood,X0,method='Nelder-Mead', tol=1e-8, bounds=bnds)

#print("calibration:",res)
print("Calibration:*******")
print("Original parameters: lambda=%1.1f, alpha=%1.1f, beta=%1.1f"%(lambda0,alpha,beta))
print("Calibrated parameters: lambda=%1.1f, alpha=%1.1f, beta=%1.1f"%(res.x[0],res.x[1],res.x[2]))


if DisplayPlots:
    plt.plot(time,lambda_t)
    plt.plot(time,events_t)
    plt.show()































