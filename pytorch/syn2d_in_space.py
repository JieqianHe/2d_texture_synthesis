from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import numpy as np
import math
import cmath
from scipy.io import loadmat
import sys
sys.path.insert(0, '/mnt/home/hejieqia/research/wavelet_functions')
from wavelet_2d import *
import os

pi = math.pi
parser = argparse.ArgumentParser()
parser.add_argument('--test_id', type=int, required=True, help='id of current test')
parser.add_argument('--n', type=int, default=128, help='size of the texture that wanted to synthesize')
parser.add_argument('--image_id', type=int, required=True, help='id of the texture that wanted to synthesize')
parser.add_argument('--min_error', type=float, default=1e-8, help='relative error for stopping criteria')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--nit', type=int, default=2000, help='iteration interval to plot')
parser.add_argument('--max_it', type=int, default=10000000, help='maximum iterations to run')
parser.add_argument('--err_it', type=int, default=500, help='iteration interval to append error and save')
parser.add_argument('--initial_type', type=str, default='uniform', help='type of initial noise')
parser.add_argument('--isplot', type=int, default=0, help='whether to plot intermediate image')
opt = parser.parse_args()
np.set_printoptions(threshold=np.inf)
 
# read parameters
test_id=opt.test_id
n = opt.n
image_id = opt.image_id
min_error = opt.min_error
max_it = opt.max_it
lr = opt.lr
nit = opt.nit
err_it = opt.err_it
initial_type = opt.initial_type
isplot = bool(opt.isplot)

if not os.path.isdir('./result%d'%test_id):
    os.mkdir('result%d'%test_id)

class scattering_2d_1layer_cov_space(torch.nn.Module):
    def __init__(self, J, K, Q, N, sigma, zeta, eta, a, sigma_low_pass, nhat=256):
        super(scattering_2d_1layer_cov_space, self).__init__()
        self.psi = []
        for j in range(-2, J*Q):
            n = int((2**(2+j/2)+1)//2)*2 + 1
            print(n)
            psi_gabor = np.zeros((K,n,n), dtype = complex)
            for k in range(K):
                theta = (k+1/2)*math.pi/K
                psi_gabor[k] = gabor_wavelet_space_2d(n,sigma,zeta,eta,theta,a,j/Q)
                psi_gabor[k] = psi_gabor[k] / np.sqrt(np.sum(np.abs(psi_gabor[k])**2))
            psi_g = np.concatenate((np.real(psi_gabor), np.imag(psi_gabor)), 0)
            psi_radial = np.zeros(((N+2)*(N-1)//2, n, n), dtype = complex)
            count = 0
            for m in range(2, N+1):
                for l in range(m):
                    psi_hat = 2**(j//Q)*radial_wavelet_freq_2d(nhat, m, l, j/Q, a, sigma)
                    psi = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(psi_hat)))
                    psi_radial[count] = psi[(nhat-n+1)//2:(nhat-n+1)//2+n, (nhat-n+1)//2:(nhat-n+1)//2+n]
                    psi_radial[count] = psi_radial[count] / np.sqrt(np.sum(np.abs(psi_radial[count])**2))
                    count += 1        
            psi_r = np.concatenate((np.real(psi_radial), np.imag(psi_radial)), 0)
#             psi = np.concatenate((psi_gabor, psi_radial), 0)
            psi = np.expand_dims(np.concatenate((psi_g, psi_r), 0), 1)
#             psi = np.expand_dims(psi_r, 1)
            (self.psi).append(torch.from_numpy(psi).float())
            
    def forward(self, x):
        for i in range(len(self.psi)):
            s = self.psi[i].shape[2]
            if i == 0:
                y = F.conv2d(x, self.psi[i], padding = s//2)
            else:
                y = torch.cat((y, F.conv2d(x, self.psi[i], padding = s//2)), 1)
        y = F.relu(y) 
        
        y = gram_matrix(y)
        return y
    
def gram_matrix(y, dim = 2):
    y = y.flatten(start_dim = dim)
    g = torch.matmul(y, torch.transpose(y, dim, dim-1))
    return g
    
   
def plot_image(x, test_id, image_id, count, nit):
    plt.figure(figsize = (10,10))
    if len(x.shape)>2:
        plt.imshow(np.sqrt(x[:,:,0]**2 + x[:,:,1]**2))
    else:
        plt.imshow(x)
    plt.colorbar()
    plt.savefig('./result%d/syn_image%d_%d'%(test_id, image_id, count//nit))
    plt.close()
    
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
x = loadmat('./data/mydata.mat')
x = x['mydata']
m = int((x.shape[0] - n)/2)
target = x[m:(m+n), m:(m+n), image_id]
target = torch.from_numpy(target).float().reshape(1,1,n,n)
target = target / 255 - 1/2
# target.to(device)
del x

N = 4
J = 4
K = 8
Q = 2
sigma = 1.1
zeta = 1.2
eta = 0.75*pi
sigma_low_pass = 15
a = 2

syn = scattering_2d_1layer_cov_space(J, K, Q, N, sigma, zeta, eta, a, sigma_low_pass)
# syn.to(device)
ind = image_id
g_target = syn(target)
if os.path.exists('./result%d/syn_result%d.npy'%(test_id, ind)):
    print('continue solving x0')
    x0 = np.load('./result%d/syn_result%d.npy'%(test_id, ind))
    x0 = np.reshape(x0, (1,1,n,n))
    x0 = Variable(torch.from_numpy(x0).float(),requires_grad = True)
else:
    x0 = Variable(torch.rand(1,1,n,n),requires_grad = True)
# x0.to(device)
    
criterion = nn.MSELoss()
optimizer = optim.Adam([x0], lr=lr)
cur_error = 0

if os.path.exists('./result%d/syn_error%d.npy'%(test_id, ind)):
    error = np.load('./result%d/syn_error%d.npy'%(test_id, ind))
else:
    error = np.array([])

for i in range(1, max_it):
    optimizer.zero_grad()
    
    g = syn(x0)
    loss = criterion(g_target, g)
    loss.backward()
    optimizer.step()
    cur_error += loss.item()

    if i % err_it == 0:
        print(cur_error / err_it)
        error = np.append(error, cur_error / err_it)
        if not math.isnan(cur_error):
            np.save('./result%d/syn_error%d.npy'%(test_id, image_id), error)
            np.save('./result%d/syn_result%d.npy'%(test_id, image_id), x0.data.cpu().numpy().reshape(n,n))
        if i > 1000 and len(error) > 2 and (abs((error[-2] - error[-1]) / error[-2]) < min_error):
            break    
        cur_error = 0
    if isplot and i % nit == 0:
        plot_image(x0.data.cpu().numpy().reshape(n,n), test_id, ind, i, err_it)          
                          
np.save('./result%d/syn_error%d.npy'%(test_id, image_id), error)
np.save('./result%d/syn_result%d.npy'%(test_id, image_id), x0.data.cpu().numpy().reshape(n,n))              
