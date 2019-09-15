
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
import math
import cmath
from scipy.io import loadmat
import sys
import matplotlib
matplotlib.use('Agg')
import pylab as plt
sys.path.insert(0, '/mnt/home/hejieqia/research/wavelet_functions')
# import matplotlib.pyplot as plt
# sys.path.insert(0, '/Users/kejiqing/Desktop/research/wavelets')

from wavelet_2d import *
pi = math.pi

parser = argparse.ArgumentParser()
parser.add_argument('--test_id', type=int, required=True, help='id of current test')
parser.add_argument('--n', type=int, default=128, help='size of the texture that wanted to synthesize')
parser.add_argument('--image_id', type=int, default=0, help='id of the texture that wanted to synthesize')
parser.add_argument('--K', type=int, default=64, help='number of angles for defining wavelets')
parser.add_argument('--Q', type=int, default=8, help='scale intervals for defining wavelets')
parser.add_argument('--J', type=float, default=5, help='largest scale for defining wavelets')
parser.add_argument('--sigma', type=float, default=1.1, help='variance of mother wavelet')
parser.add_argument('--sigma_low_pass', type=float, default=5, help='variance of low pass')
parser.add_argument('--zeta', type=float, default=1.2, help='bias over y direction of mother wavelet')
parser.add_argument('--eta', type=float, default=0.75 * pi, help='central frequency of mother wavelet')
parser.add_argument('--a', type=int, default=2, help='parameter to dilate wavelets')
parser.add_argument('--min_error', type=float, default=1e-9, help='how much error want to reduce to relative to initial error')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--nit', type=int, default=500, help='iteration interval to save')
parser.add_argument('--err_it', type=int, default=50, help='iteration interval to append error')
parser.add_argument('--initial_type', type=str, default='uniform', help='type of initial noise')
parser.add_argument('--layer2', type=int, default=0, help='whether to do 2nd layer scattering')
parser.add_argument('--cov', type=int, default=0, help='whether to compute covariance')
parser.add_argument('--eps', type=float, default=0.001, help='epsilon in barycenter algorithm')
opt = parser.parse_args()
 
# read parameters
print('initializing parameters...')
test_id=opt.test_id
n = opt.n
Q = opt.Q
J = opt.J
K = opt.K
a = opt.a
sigma = opt.sigma
sigma_low_pass = opt.sigma_low_pass
zeta = opt.zeta
eta = opt.eta
min_error = opt.min_error
lr = opt.lr
nit = opt.nit
err_it = opt.err_it
initial_type = opt.initial_type
layer2 = bool(opt.layer2)
cov = bool(opt.cov)
eps = opt.eps

def layer1_ground_distance(K, J, Q, nnorm = 2):
    n = int(K * (J * Q + 1))
    
    g = np.zeros((n,n))
    x = np.arange(0,J*Q + 1, 1)
    y = np.arange(K)
    
    X,Y = np.meshgrid(x,y)
    X,Y = X.reshape((-1,1)), Y.reshape((-1,1))
    ind = np.concatenate((X,Y), 1)
    
    ind1 = np.expand_dims(ind, 0)
    ind2 = np.expand_dims(ind, 1)
    
    dscale = np.abs(ind1[:,:,0] - ind2[:,:,0])
    dangle1 = np.expand_dims(np.abs(ind1[:,:,1] - ind2[:,:,1]), 2)
    dangle2 = K - dangle1
    dangle = np.min(np.concatenate((dangle1,dangle2), 2), 2)
    
    g = dscale**nnorm + dangle**nnorm
    
    return g
    
def wavelet_euc_axis(K,J,Q):
    # K : number of angles in half plane
    x = np.zeros((K,int(J*Q+1),2))
    for k in range(K):
        for j in range(int(J*Q+1)):
            r = 2**(-j/Q)
            a = pi*k / K
            x[k,j,0] = r*np.cos(a)
            x[k,j,1] = r*np.sin(a)
    return x

def euc_to_polar(x,J,K):
    # transfer euclidean to polar axis with log scale
    y = np.zeros(x.shape)
    y[0,0] = - J - 1
    y[1:,0] = np.log2(np.sqrt(x[1:,0]**2 + x[1:,1]**2))
    y[:,1] = np.arctan2(x[:,1],x[:,0])
    ind = np.where(y[:,1] < 0)
    y[ind,1] = y[ind,1] + math.pi
    y[:,1] = y[:,1] * K / math.pi
    
    return y

def scat_stat_polar_axis_layer2(K,J,Q):
    x_wave = wavelet_euc_axis(K,J,Q)
    x_low = np.zeros((1,2))
    ax = x_low
    for j1 in range(int(J*Q)):
        ax = np.concatenate((ax, x_wave[:,j1,:]),  0)
        for k1 in range(K):
            x_1 = x_wave[k1,j1,:]
            x_2 = x_wave[:,(j1+1):,:]
            x_temp  = x_1 + x_2
            ax = np.concatenate((ax, np.reshape(x_temp, (-1,2))), 0)
    ax_euc = np.concatenate((ax, x_wave[:,-1,:]),0)
    ax_polar = euc_to_polar(ax_euc,J,K)
    return ax_euc, ax_polar


def layer2_ground_distance(K,J,Q, nnorm = 2):
    ax_euc, ax_polar = scat_stat_polar_axis_layer2(K,J,Q)
    n = ax_polar.shape[0]
    r = ax_polar[:,0]
    a = ax_polar[:,1]
    
    r1 = np.expand_dims(r,0)
    r2 = np.expand_dims(r,1)
    dscale = np.abs(r1 - r2)
    
    a1 = np.expand_dims(a,0)
    a2 = np.expand_dims(a,1)
    dangle1 = np.expand_dims(np.abs(a1 - a2),2)
    dangle2 = K - dangle1
    dangle = np.min(np.concatenate((dangle1,dangle2), 2), 2)
    
    g = np.zeros((n,n))
    g = dscale**nnorm + dangle**nnorm
    return g

class scattering_2d_1layer(torch.nn.Module):
    def __init__(self, phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag):
        super(scattering_2d_1layer, self).__init__()
        self.phi_hat_real = phi_hat_real # n * n
        self.phi_hat_imag = phi_hat_imag # n * n
        self.psi_hat_real = psi_hat_real # K * J * n * n
        self.psi_hat_imag = psi_hat_imag # K * J * n * n
        print('scattering type: 1 layer')
 
    def forward(self, x_hat):
        a = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) *
                                       (self.phi_hat_real**2 + self.phi_hat_imag**2), 0), 0)
        a = a.unsqueeze(0) 
        b = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) *
                                       (self.psi_hat_real**2 + self.psi_hat_imag**2), 3), 2)
        s = torch.cat((a, b.flatten()), 0)
        return s
    
class scattering_2d_2layers(torch.nn.Module):
    def __init__(self, phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag, second_all = False):
        super(scattering_2d_2layers, self).__init__()
        self.phi_hat_real = phi_hat_real # n * n
        self.phi_hat_imag = phi_hat_imag # n * n
        self.psi_hat_real = psi_hat_real # K * J * n * n
        self.psi_hat_imag = psi_hat_imag # K * J * n * n
        self.second_all = second_all # whether to do all pair of (j1, j2) at second layer or just j2 > j1
        print('scattering type: 2 layers.')

    def forward(self, x_hat):
        #K = int(self.psi_hat_real.size()[0]/2)
        # x_hat: n * n * 2
        s = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) * (self.phi_hat_real**2 + self.phi_hat_imag**2), 0), 0)
        s = s.unsqueeze(0)
        J = self.psi_hat_real.size()[1]
        for i in range(J):
            temp_real = x_hat[:,:,0] * self.psi_hat_real[:,i,:,:] - x_hat[:,:,1] * self.psi_hat_imag[:,i,:,:]
            temp_imag = x_hat[:,:,0] * self.psi_hat_imag[:,i,:,:] + x_hat[:,:,1] * self.psi_hat_real[:,i,:,:]
            temp = torch.ifft(torch.cat((temp_real.unsqueeze(3), temp_imag.unsqueeze(3)), 3), 2) # K * n * n * 2

            temp2 = torch.rfft(torch.sqrt(temp[:,:,:,0]**2 + temp[:,:,:,1]**2 + 1e-8), 2, onesided = False) # K * n * n * 2

            a = 1/(2 * pi)**2 * torch.mean(torch.mean((temp2[:,:,:,0]**2 + temp2[:,:,:,1]**2) * (self.phi_hat_real**2 + self.phi_hat_imag**2), 2), 1)
            s = torch.cat((s, a), 0)
            if i < J - 1:
                temp3 = (temp2[:,:,:,0]**2 + temp2[:,:,:,1]**2).unsqueeze(1).unsqueeze(2)
                if self.second_all:
                    temp4 = (self.psi_hat_real[:,:,:,:]**2 + self.psi_hat_imag[:,:,:,:]**2).unsqueeze(0)
                else:
                    temp4 = (self.psi_hat_real[:,(i+1):J,:,:]**2 + self.psi_hat_imag[:,(i+1):J,:,:]**2).unsqueeze(0)
                b = 1/(2 * pi)**2 * torch.mean(torch.mean(temp3 * temp4, 4), 3)
                s = torch.cat((s, b.flatten()), 0)

        return s

def initial_scat(n, K, J, Q, sigma, sigma_low_pass, zeta, eta, a, layer2):
    psi_hat = gabor_wavelet_family_freq_2d(n, K, J, Q, sigma, zeta, eta, a)
    phi_hat = low_pass_freq(n, sigma_low_pass, sigma_low_pass)
#     phi_hat, psi_hat = LP_wavelet_family_freq_2d(n,J,Q,K)

    psi_hat = np.swapaxes(psi_hat, 0, 2)
    psi_hat = np.swapaxes(psi_hat, 1, 3) # K * J * n * n
    phi_hat_real = torch.from_numpy(np.real(np.fft.fftshift(phi_hat, axes = (0,1)))).float()
    phi_hat_imag = torch.from_numpy(np.imag(np.fft.fftshift(phi_hat, axes = (0,1)))).float()
    psi_hat_real = torch.from_numpy(np.real(np.fft.fftshift(psi_hat, axes = (2,3)))).float()
    psi_hat_imag = torch.from_numpy(np.imag(np.fft.fftshift(psi_hat, axes = (2,3)))).float()
    if torch.cuda.is_available():
        psi_hat_real = psi_hat_real.cuda()
        psi_hat_imag = psi_hat_imag.cuda()  
        phi_hat_real = phi_hat_real.cuda()
        phi_hat_imag = phi_hat_imag.cuda()
    # initialize scattering module
    if not layer2:
        scat = scattering_2d_1layer(phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag)
    else:
        scat = scattering_2d_2layers(phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag)
    return scat

def low_pass_freq(n, sigma1, sigma2):
    pi = math.pi
    omega1 = (np.arange(n) - n/2) * 2 * pi / n
    omega2 = omega1

    omega1, omega2 = np.meshgrid(omega1,omega2)
    phi_hat = np.exp(- sigma1**2 * omega1**2 / 2 - sigma2**2 * omega2**2 / 2)
    return phi_hat

def plot_image(x, test_id, image_id, count, nit):
    plt.figure(figsize = (10,10))
    if len(x.shape)>2:
        plt.imshow(np.sqrt(x[:,:,0]**2 + x[:,:,1]**2))
    else:
        plt.imshow(x)
    plt.colorbar()
    plt.savefig('./result%d/syn_image%d_%d'%(test_id, image_id, count//nit))
    plt.close()
    
def WassersteinBarycenter(C, pk, lamb, nit = 100000, eps = 1/500, tol_dif = 1e-3):
    # C: ground distance, d * d
    # pk: N histogram of size d, d * N
    # lamb: normalized weight, N * 1
    print('looking for barycenter...')
    ita = np.exp(- C / eps) # d * d
    ita[np.where(ita < 1e-300)]= 1e-300
    
    count = 0
     
    uitav = np.matmul(ita, pk / np.expand_dims(np.sum(ita, 0),1)) # d * N
    u = np.exp(np.matmul(np.log(uitav), lamb)) / uitav # d * 1

    differ = np.sum(np.std(uitav, 1))
    dif0 = np.sum(np.std(uitav, 1))

    while (count < nit) & (differ/dif0 > tol_dif):
        uitav = u * np.matmul(ita, pk / np.matmul(np.transpose(ita), u))
        u = u * np.exp(np.matmul(np.log(uitav), lamb)) / uitav
        differ = np.sum(np.std(uitav, 1))
        
        if count % 500 == 0:
            print('count: ', count) 
            print(differ)
        count += 1
    return np.mean(uitav, 1)


def synthesis(s_target, test_id, ind, scat, n, min_error, err_it, nit, is_complex = False, initial_type = 'gaussian'):
    print('shape of s:', s_target.size())
#     s_target = s_target / torch.sum(s_target) 
    if initial_type == 'gaussian':
        x0 = torch.randn(n,n)
    elif initial_type == 'uniform':
        x0 = torch.rand(n,n)

    if torch.cuda.is_available():
        s_target = s_target.cuda()
        x0 = x0.cuda()
    x0 = Variable(x0, requires_grad=True)
    
    x0_hat = torch.rfft(x0, 2, onesided = False)
    
    s0 = scat(x0_hat)
    loss = nn.MSELoss()
    optimizer = optim.Adam([x0], lr=lr)
    output = loss(s_target, s0)
    l0 = output
    error = []
    count = 0
    while output / l0 > min_error:
        optimizer.zero_grad() 
        x0_hat = torch.rfft(x0, 2, onesided = False)
        s0 = scat(x0_hat)
        output = loss(s_target, s0)
        if count % err_it ==0:   
            error.append(output.item())
        output.backward()
        optimizer.step()
        output = loss(s_target, s0)
        if count % nit == 0:
            print(output.data.cpu().numpy())
            np.save('./result%d/syn_error%d.npy'%(test_id, ind), np.asarray(error))
            np.save('./result%d/syn_result%d.npy'%(test_id, ind), x0.data.cpu().numpy())
            # plot_image(x0.data.cpu().numpy(), test_id, ind, count, nit)
        count += 1
    print('error reduced by: ', output / l0)
    print('error supposed reduced by: ', min_error)
    np.save('./result%d/syn_error%d.npy'%(test_id, ind), np.asarray(error))
    np.save('./result%d/syn_result%d.npy'%(test_id, ind), x0.data.cpu().numpy())

# initialize scattering
scat = initial_scat(n, K, J, Q, sigma, sigma_low_pass, zeta, eta, a, layer2)

# extract two inputs    
id_all = [18,19]    
# load data
# x = loadmat('/Users/kejiqing/Desktop/research/wavelet&EMD/code/images/mydata.mat')
x = loadmat('./data/mydata.mat')
x = x['mydata']
x_data = torch.from_numpy(x).float()
m = int((x.shape[0] - n)/2)
x_data = x_data[m:(m+n), m:(m+n), id_all]
del x

if torch.cuda.is_available():
    x_data = x_data.cuda()
    scat = scat.cuda()

# compute scattering stat
target_hat1 = torch.rfft(x_data[:,:,0], 2, onesided = False)
s1 = scat(target_hat1)
target_hat2 = torch.rfft(x_data[:,:,1], 2, onesided = False)
s2 = scat(target_hat2)
print('s1 size: ', s1.size())
# find scattering barycenter
print('looking for barycenter...')
if not layer2:
    g = layer1_ground_distance(K, J, Q)
else:
    g = layer2_ground_distance(K, J, Q)
    g = g[1:,1:]
g = g/np.median(g)
print('g size: ', g.shape)
p10 = s1[:1].data.cpu().numpy().reshape(-1,1)
p20 = s2[:1].data.cpu().numpy().reshape(-1,1)
p1 = s1[1:].data.cpu().numpy().reshape(-1,1)
p2 = s2[1:].data.cpu().numpy().reshape(-1,1)

p10 = p10 / np.sum(p1) 
p20 = p20 / np.sum(p2)
p1 = p1 / np.sum(p1) # transform into probability distribution
p2 = p2 / np.sum(p2)

pk = np.concatenate((p1, p2), 1)
lamb = np.ones(2).reshape(2,1)/2
res0 = (p10 + p20) / 2
print('p0_middle: ', res0)

res = WassersteinBarycenter(g, pk, lamb, eps = eps, tol_dif = 1e-6)

p1_numpy = np.concatenate((p10.reshape(1,1), p1.reshape(-1,1)), 0)
p2_numpy = np.concatenate((p20.reshape(1,1), p2.reshape(-1,1)), 0)
s_target = np.concatenate((res0.reshape(1,1), res.reshape(-1,1)), 0)

np.save('./result%d/s11.npy'%test_id, p1_numpy)
np.save('./result%d/s21.npy'%test_id, p2_numpy)
np.save('./result%d/s_bc1.npy'%test_id, s_target)

s1 = torch.from_numpy(p1_numpy).float()
s2 = torch.from_numpy(p2_numpy).float()
s_bc = torch.from_numpy(s_target).float()

# downsampling the scattering statistics for synthesis
# p1 = p1.reshape(K, int(J*Q+1))
# p2 = p2.reshape(K, int(J*Q+1))
# res = res.reshape(K, int(J*Q+1))

# K1 = 16
# J1 = 5
# Q1 = 4
# p1_ = np.zeros((K1,J1*Q1 + 1))
# p2_ = np.zeros((K1,J1*Q1 + 1))
# p_bc_ = np.zeros((K1,J1*Q1 + 1))
# for k in range(K1):
#     for j in range(J1*Q1+1):
#         p1_[k,j] = p1[8*k, 4*j]
#         p2_[k,j] = p2[8*k, 4*j]
#         p_bc_[k,j] = res[8*k, 4*j]
# s1 = torch.from_numpy(np.concatenate((p10.reshape(1,1), p1_.reshape(-1,1)), 0)).float()
# s2 = torch.from_numpy(np.concatenate((p20.reshape(1,1), p2_.reshape(-1,1)), 0)).float()
# s_bc = torch.from_numpy(np.concatenate((res0.reshape(1,1), p_bc_.reshape(-1,1)), 0)).float()
s1 = s1 * n**2
s2 = s2 * n**2
s_bc = s_bc * n**2

# scat1 = initial_scat(n, K1, J1, Q1, sigma, sigma_low_pass, zeta, eta, a, layer2)

if torch.cuda.is_available():
#     scat1 = scat1.cuda()
    s1 = s1.cuda()
    s2 = s2.cuda()
    s_bc = s_bc.cuda()
# np.save('./result%d/s1.npy'%test_id, s1.data.cpu().numpy())
# np.save('./result%d/s2.npy'%test_id, s2.data.cpu().numpy())
# np.save('./result%d/s_bc.npy'%test_id, s_bc.data.cpu().numpy())

print('synthesizing image ' + str(id_all[0]))
synthesis(s1.squeeze(), test_id, id_all[0], scat, n, min_error, err_it, nit, initial_type = initial_type, is_complex = False)
print('synthesizing image ' + str(id_all[1]))
synthesis(s2.squeeze(), test_id, id_all[1], scat, n, min_error, err_it, nit, initial_type = initial_type, is_complex = False)
print('synthesizing image barycenter')
synthesis(s_bc.squeeze(), test_id, 1819, scat, n, min_error, err_it, nit, initial_type = initial_type, is_complex = False)

