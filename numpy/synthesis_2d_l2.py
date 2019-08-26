import numpy as np
import math
import cmath
#import matplotlib
#matplotlib.use('Agg')
#import pylab as plt
from scipy import signal
from scipy.optimize import minimize
from scipy.io import loadmat
import random
from numpy import linalg as LA
import time
# import imageio
import argparse

import sys
sys.path.insert(0, '/mnt/home/hejieqia/research/wavelet_functions')
from wavelet_2d import *

x = loadmat('mydata.mat')
x = x['mydata']
#x : 640 x 640 x 111
pi = math.pi

parser = argparse.ArgumentParser()
parser.add_argument('--test_id', type=int, required=True, help='id of current test')
parser.add_argument('--n', type=int, default=64, help='size of the texture that wanted to synthesize')
parser.add_argument('--image_id', type=int, default=0, help='id of the texture that wanted to synthesize')
parser.add_argument('--K', type=int, default=6, help='number of angles for defining wavelets')
parser.add_argument('--Q', type=int, default=2, help='scale intervals for defining wavelets')
parser.add_argument('--J', type=int, default=4, help='largest scale for defining wavelets')
parser.add_argument('--sigma', type=int, default=1.1, help='variance of mother wavelet')
parser.add_argument('--zeta', type=int, default=1.2, help='bias over y direction of mother wavelet')
parser.add_argument('--eta', type=int, default=0.75 * pi, help='central frequency of mother wavelet')
parser.add_argument('--a', type=int, default=2, help='parameter to dilate wavelets')
parser.add_argument('--jacob', type=bool, default=True, help='whether to specify jacobian')
parser.add_argument('--max_err', type=int, default=1e-6, help='maximum loss to stop optimization')
parser.add_argument('--max_epoch', type=int, default=1, help='maximum number of epochs')
parser.add_argument('--pad', type=bool, default=False, help='whether to pad image during optimization')
opt = parser.parse_args()

# read parameters
test_id=opt.test_id
n = opt.n
image_id = opt.image_id
Q = opt.Q
J = opt.J
K = opt.K
a = opt.a
sigma = opt.sigma
zeta = opt.zeta
eta = opt.eta
jacob = opt.jacob
max_err = opt.max_err
max_epoch = opt.max_epoch
pad = opt.pad

def scat_2d_l2(x_hat, psi_hat):
    # compute l2 normm of scattering coefficients
    pi = math.pi
    s = 1/(2 * pi)**2 * np.mean(np.abs(x_hat)**2 * np.abs(psi_hat)**2, axis = 0)
    return s

def diff_l2(y0_hat, sx, psi_hat, n, index1, index2):
    # difference vector between first moment wavelet coefficients
    y0_hat = np.expand_dims(y0_hat, axis = 1)
    y_hat = y0_hat[0:n, :] + 1j * y0_hat[n:, :]
    sy = scat_2d_l2(y_hat, psi_hat)
    diff = np.mean((sx - sy)**2)
    print('diff: ', diff)
    return diff

def synthesis_l2(x, psi_hat, jacob, max_err, max_epoch, *args):
    # synthesis 2d signal x with greedy algorithm
    
    # collect parameter
    nx = x.shape
    npsi = psi.shape 
    
    psi_hat = np.reshape(psi_hat, (npsi[0]*npsi[1], npsi[2]*npsi[3]))
    nw = npsi[2]*npsi[3]
    x_hat = np.reshape(np.fft.fftshift(np.fft.fft2(x)), (nx[0]*nx[1], 1))
    sx = scat_2d_l2(x_hat, psi_hat)
    
    
    # initialize test signal
    if len(args) == 0:
        y_hat = np.random.random(2*nx[0]*nx[1]) # initialize y
    else:
        y_hat = args[0]
        
    y = np.zeros((nx[0], nx[1], 1))
    
    err = 1
    epoch = 0
    tic = time.time()
    print('initial loss: ', diff_l2(y_hat, sx, psi_hat, nx[0]*nx[1]))
    while (err > max_err) & (epoch < max_epoch):
        
        epoch += 1
        ind = np.random.choice(nw, nw, replace = False) # randomize index of wavelets
        print('epoch:', epoch)
        
        ind = []
        ind0 = np.arange(nw // nrun)
        for i in range(nrun):
            ind.append(ind0 * nrun + i)
            print('current loss:', diff_l2(y_hat, sx[ind], psi_hat[:,ind], nx[0]*nx[1]))
            if jacob:
                res = minimize(diff_l2, y_hat, args = (sx[ind], psi_hat[:,ind], nx[0]*nx[1]),
                               jac = jac_l2, method='BFGS')
            else:
                res = minimize(diff_l2, y_hat, args = (sx[ind], psi_hat[:,ind], nx[0]*nx[1]),
                               method='BFGS')
            y0_hat = np.reshape(res.x, (nx[0], nx[1], 2))
            y0_hat = y0_hat[:,:,0] + 1j * y0_hat[:,:,1]
            y0 = np.fft.ifft2(np.fft.fftshift(y0_hat))
            y = np.append(y, np.reshape(y0, (nx[0], nx[1], 1)), axis = 2)

            err = res.fun
            print('optimized loss:', err)
    toc = time.time()
    print('running time:', toc - tic)
    return y

def jac_l2(y0_hat, sx, psi_hat, n2, index1, index2):
    # jacobian function for difference
    y0_hat = np.expand_dims(y0_hat, axis = 1)
    pi = math.pi
    nw = psi_hat.shape[1]
    y_hat = y0_hat[0:n2, :] + 1j * y0_hat[n2:, :]
    sy = scat_2d_l2(y_hat, psi_hat) # compute scattering
    
    g = np.zeros(2 * n2)
    temp1 = np.squeeze(2 * y0_hat)
    psi_hat_temp = np.concatenate((psi_hat, psi_hat), 0)
    for j in range(nw):
        temp2 = 1/(2 * pi)**2 * np.abs(psi_hat_temp[:, j])**2 * temp1
        g = g + 2 * (sy[j] - sx[j]) * temp2
        
    g[index1] = g[index1] + g[index2]
    g[index2] = np.copy(g[index1])
    g[index1 + n2] = g[index1 + n2] - g[index2 + n2]
    g[index2 + n2] = np.copy(- g[index1 + n2])
    g[int(n2 + n2 / 2 + np.sqrt(n2) / 2)] = 0
    return g / n2

def index_transform(ind1, ind2, n):
    nind = ind1.shape[0]
    ind = []
    for i in range(nind):
        for j in range(nind):
            ind.append(ind1[i] * n + ind2[j])
    return ind

def find_index(n):
    ind_all = np.arange(n**2)
    ind1 = np.where(ind_all // n > 0)[0]
    ind2 = np.where(ind_all % n > 0)[0]
    ind3 = np.where(ind_all < n**2/2 + n/2)[0]
    ind4 = np.where(ind_all > n**2/2 + n/2)[0]
    index1 = np.intersect1d(ind1,ind2)
    index1 = np.intersect1d(index1,ind3)
    index2 = np.intersect1d(ind2, ind4)
    return index1, np.flip(index2, axis = 0)

def initialize_y_hat(n):
    y_hat = np.random.random(2*n**2) # initialize y
    index1, index2 = find_index(n)
    y_hat[index2] = np.copy(y_hat[index1])
    y_hat[index2 + n**2] = np.copy(- y_hat[index1 + n**2])
    y_hat[int(n**2 + n**2/2 + n/2)] = 0
    return y_hat

target = x[0:n, 0:n, image_id]
del x

# define wavelets
psi = gabor_wavelet_family_space_2d(n, K, Q, J, sigma, zeta, eta, a)
if pad:
    psi_hat = gabor_wavelet_family_freq_2d(3 * n, K, J, Q, sigma, zeta, eta, a)
else:
    psi_hat = gabor_wavelet_family_freq_2d(n, K, J, Q, sigma, zeta, eta, a)

x = target
# collect parameter
npsi = psi.shape 
psi_hat = np.reshape(psi_hat, (n**2, npsi[2]*npsi[3]))
nw = npsi[2]*npsi[3]
x_hat = np.reshape(np.fft.fftshift(np.fft.fft2(x)), (n**2, 1))
sx = scat_2d_l2(x_hat, psi_hat)

# initialize test signal
y_hat = initialize_y_hat(n)
index1, index2 = find_index(n)

y = np.zeros((2*n**2, 1))
error = []
err = 1
epoch = 0
tic = time.time()
print('initial loss: ', diff_l2(y_hat, sx, psi_hat, n**2, index1, index2))
while (err > max_err) & (epoch < max_epoch):

    epoch += 1
    print('epoch:', epoch)
       
    index_interval = npsi[2]
    while index_interval > 1:
        index_interval = int(index_interval / 2)
        ind1 = np.arange(0, npsi[2], index_interval)
        ind2 = np.copy(ind1)
        ind = index_transform(ind1, ind2, npsi[2])
        print('wavelet index: ', ind)
        print('current loss:', diff_l2(y_hat, sx[ind], psi_hat[:,ind], n**2, index1, index2))
        if jacob:
            res = minimize(diff_l2, y_hat, args = (sx[ind], psi_hat[:,ind], n**2, index1, index2),
                           jac = jac_l2, method='BFGS')
        else:
            res = minimize(diff_l2, y_hat, args = (sx[ind], psi_hat[:,ind], n**2, index1, index2),
                           method='BFGS')
        y_hat = res.x
        #y0_hat = np.reshape(res.x, (n, n, 2))
        #y0_hat = y0_hat[:,:,0] + 1j * y0_hat[:,:,1]
        #y0 = np.fft.ifft2(np.fft.fftshift(y0_hat))
        #y = np.append(y, np.reshape(y0, (n, n, 1)), axis = 2)
        y = np.append(y, np.reshape(y_hat, (2*n**2, 1)), axis = 1)
        np.save('./result/synthesis_l2_%s.npy'%test_id, y)
        err = res.fun
        error.append(err)
        np.save('./result/synthesis_l2_error_%s.npy'%test_id, np.asarray(error))
        print('optimized loss:', err)
toc = time.time()
print('running time:', toc - tic)
# np.save('./result/synthesis_l2_%s.npy'%test_id, res)
