import numpy as np
import math
import cmath
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from scipy import signal
from scipy.optimize import minimize
from scipy.io import loadmat
import random
from numpy import linalg as LA
import time
import imageio
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
parser.add_argument('--jacob', type=int, default=True, help='whether to specify jacobian')
parser.add_argument('--max_err', type=int, default=1e-6, help='maximum loss to stop optimization')
parser.add_argument('--max_epoch', type=int, default=1, help='maximum number of epochs')
parser.add_argument('--pad', type=int, default=False, help='whether to pad image during optimization')
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

def diff(y0, sx, psi_hat, nx, psi, pad):
    # difference vector between first moment wavelet coefficients
    y = np.reshape(y0, nx)
    sy = scat_coeff_2d(y, psi_hat)
    diff = np.sum((sx - sy)**2)
    return diff


def synthesis(x, psi_hat, psi, jacob, pad, max_err, max_epoch, test_id, *args):
    # synthesis 2d signal x with greedy algorithm
    
    # collect parameter
    nx = x.shape
    npsi = psi.shape 
    
    psi = np.reshape(psi, (npsi[0], npsi[1], npsi[2]*npsi[3], 1))
    psi_hat = np.reshape(psi_hat, (npsi[0], npsi[1], npsi[2]*npsi[3], 1))
    nw = npsi[2]*npsi[3]
    sx = scat_coeff_2d(x, psi_hat)
    
    
    # initialize test signal
    if len(args) == 0:
        y0 = np.random.random(nx[0]*nx[1])  # initialize y
    else:
        y0 = args[0]
        
    y = np.zeros((nx[0], nx[1], 1))
    
    err = 1
    epoch = 0
    tic = time.time()
    print('initial loss: ', diff(y0, sx, psi_hat, nx, psi, pad))
    while (err > max_err) & (epoch < max_epoch):
        
        epoch += 1
        ind = np.random.choice(nw, nw, replace = False) # randomize index of wavelets
        print('epoch:', epoch)
        
        for i in range(nw):
            print('current loss:', diff(y0, sx[np.append([0], ind[0:i+1] + 1)], psi_hat[:,:,ind[0:i+1],:], \
                                                 nx, psi[:,:,ind[0:i+1],:], pad))
            if jacob:
                res = minimize(diff, y0, args = (sx[np.append([0], ind[0:i+1] + 1)], psi_hat[:,:,ind[0:i+1],:], \
                                                 nx, psi[:,:,ind[0:i+1],:], pad), \
                               jac = jac, method='BFGS')
            else:
                res = minimize(diff, y0, args = (sx[np.append([0], ind[0:i+1] + 1)], psi_hat[:,:,ind[0:i+1],:], \
                                                 nx, psi[:,:,ind[0:i+1],:], pad), \
                               method='BFGS')
            y0 = res.x
            y = np.append(y, np.reshape(y0, (nx[0], nx[1], 1)), axis = 2)
            np.save('synthesized%s.npy'%test_id, y)
            err = res.fun
            print('optimized loss:', err)
    toc = time.time()
    print('running time:', toc - tic)
    return y

def jac(y0, sx, psi_hat, nx, psi, pad):
    # jacobian function for difference
    epsilon = 1e-6
    n = nx[0]
    nw = psi_hat.shape[2]
    y = np.reshape(y0, nx) # reshape
    sy = scat_coeff_2d(y, psi_hat) # compute scattering
    y_hat = np.fft.fft2(np.fft.fftshift(y)) # Fourier transform of y
    
    g = np.zeros(n**2)
    g = g + 2 * (sy[0] - sx[0]) * np.ones(n**2)
    
    for i in range(nw):
        temp1 = np.fft.fftshift(np.fft.ifft2(np.multiply(y_hat, np.fft.fftshift(psi_hat[:,:,i,0]))))
        temp_real = np.reshape(np.divide(np.real(temp1), abs(temp1) + epsilon), n*n)
        temp_imag = np.reshape(np.divide(np.imag(temp1), abs(temp1) + epsilon), n*n)
        psi_shift = np.zeros((n,n,n,n), dtype = complex)
        for p in range(n):
            for q in range(n):
                psi_shift[:,:,p,q] = np.roll(psi[:,:,i,0], (p, q), axis = (0,1))
        psi_shift = np.reshape(psi_shift, (n*n, n*n))
        temp2 = np.matmul(temp_real, np.real(psi_shift)) + np.matmul(temp_imag, np.imag(psi_shift))
        g = g + 2 * (sy[i+1] - sx[i+1]) * temp2
        del psi_shift
    return g / n**2

# define signal
target = x[0:n, 0:n, image_id]
del x

# define wavelets
psi = gabor_wavelet_family_space_2d(n, K, Q, J, sigma, zeta, eta, a)
if pad:
    psi_hat = gabor_wavelet_family_freq_2d(3 * n, K, J, Q, sigma, zeta, eta, a)
else:
    psi_hat = gabor_wavelet_family_freq_2d(n, K, J, Q, sigma, zeta, eta, a)

res = synthesis(target, psi_hat, psi, jacob, pad, max_err, max_epoch, test_id)

np.save('./result/synthesized%s.npy'%test_id, res)
