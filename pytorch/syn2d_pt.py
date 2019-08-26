from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
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
sys.path.insert(0, '/mnt/home/hejieqia/research/wavelet_functions')
from wavelet_2d import *

pi = math.pi
parser = argparse.ArgumentParser()
parser.add_argument('--test_id', type=int, required=True, help='id of current test')
parser.add_argument('--n', type=int, default=128, help='size of the texture that wanted to synthesize')
parser.add_argument('--image_id', type=int, default=0, help='id of the texture that wanted to synthesize')
parser.add_argument('--K', type=int, default=8, help='number of angles for defining wavelets')
parser.add_argument('--Q', type=int, default=2, help='scale intervals for defining wavelets')
parser.add_argument('--J', type=float, default=5, help='largest scale for defining wavelets')
parser.add_argument('--sigma', type=float, default=1.1, help='variance of mother wavelet')
parser.add_argument('--sigma_low_pass', type=float, default=15, help='variance of low pass')
parser.add_argument('--zeta', type=float, default=1.2, help='bias over y direction of mother wavelet')
parser.add_argument('--eta', type=float, default=0.75 * pi, help='central frequency of mother wavelet')
parser.add_argument('--a', type=int, default=2, help='parameter to dilate wavelets')
parser.add_argument('--min_error', type=float, default=1e-8, help='how much error want to reduce to relative to initial error')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--nit', type=int, default=2000, help='iteration interval to save')
parser.add_argument('--err_it', type=int, default=50, help='iteration interval to append error')
parser.add_argument('--initial_type', type=str, default='uniform', help='type of initial noise')
parser.add_argument('--layer2', type=int, default=1, help='whether to do 2nd layer scattering')
parser.add_argument('--cov', type=int, default=0, help='whether to compute covariance')
opt = parser.parse_args()
 
# read parameters
test_id=opt.test_id
n = opt.n
id_all = opt.image_id
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
texture = True
navier_stokes = False


class scattering_2d(torch.nn.Module):
    def __init__(self, psi_hat_real, psi_hat_imag):
        super(scattering_2d, self).__init__()
        self.psi_hat_real = psi_hat_real
        self.psi_hat_imag = psi_hat_imag
        print('scattering type: 1 layer')

    def forward(self, x_hat):
        s = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0:1]**2 + x_hat[:,:,1:2]**2) *
                                       (self.psi_hat_real**2 + self.psi_hat_imag**2), 0), 0)
        return s

class scattering_2d_cov(torch.nn.Module):
    def __init__(self, phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag):
        super(scattering_2d_cov, self).__init__()
        self.phi_hat_real = phi_hat_real # n * n
        self.phi_hat_imag = phi_hat_imag # n * n
        self.psi_hat_real = psi_hat_real # K * J * n * n
        self.psi_hat_imag = psi_hat_imag # K * J * n * n
        print('scattering type: covariance')
 
    def forward(self, x_hat):
        a = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) *
                                       (self.phi_hat_real**2 + self.phi_hat_imag**2), 0), 0)
        a = a.unsqueeze(0) 
        b = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) *
                                       (self.psi_hat_real**2 + self.psi_hat_imag**2), 3), 2)
        s = torch.cat((a, b.flatten()), 0)
        
        K = self.psi_hat_real.size()[0]
        J = self.psi_hat_real.size()[1]
        
        for i in range(1,4):
            c1 = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) *
                                           (self.psi_hat_real[:,0:(J-i),:,:] * self.psi_hat_real[:,i:J,:,:] + 
                                            self.psi_hat_imag[:,0:(J-i),:,:] * self.psi_hat_imag[:,i:J,:,:]), 3), 2)
            c2 = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) *
                                           (self.psi_hat_imag[:,0:(J-i),:,:] * self.psi_hat_real[:,i:J,:,:] - 
                                            self.psi_hat_real[:,0:(J-i),:,:] * self.psi_hat_imag[:,i:J,:,:]), 3), 2)
            c = c1**2 + c2**2
            s = torch.cat((s, c.flatten()), 0)
        
        for i in range(1,4):
            d1 = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) *
                                           (self.psi_hat_real[0:(K-i),:,:,:] * self.psi_hat_real[i:K,:,:,:] + 
                                            self.psi_hat_imag[0:(K-i),:,:,:] * self.psi_hat_imag[i:K,:,:,:]), 3), 2)
            d2 = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) *
                                           (self.psi_hat_imag[0:(K-i),:,:,:] * self.psi_hat_real[i:K,:,:,:] - 
                                            self.psi_hat_real[0:(K-i),:,:,:] * self.psi_hat_imag[i:K,:,:,:]), 3), 2)
            d = d1**2 + d2**2
            s = torch.cat((s, d.flatten()), 0)
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
        # x_hat: n * n * 2
        s = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) * (self.phi_hat_real**2 + self.phi_hat_imag**2), 0), 0)
        s = s.unsqueeze(0)
        J = psi_hat_real.size()[1]
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

def synthesis(target, test_id, ind, scat, n, min_error, err_it, nit, is_complex = False, initial_type = 'gaussian'):
    if torch.cuda.is_available():
        target = target.cuda()
    # set up target
    target_ = n**2 * target / torch.sqrt(torch.sum(target**2))

    target_hat = torch.rfft(target_, 2, onesided = False)
    s_target = scat(target_hat)
    print('shape of s:', s_target.size())
#    s_target = s_target / torch.sum(s_target) 

    if initial_type == 'gaussian':
        x0 = torch.randn(n,n)
    elif initial_type == 'uniform':
        x0 = torch.rand(n,n)
    x0 = n**2 * x0/torch.sqrt(torch.sum(target**2))

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

# define wavelets
psi_hat = gabor_wavelet_family_freq_2d(n, K, J, Q, sigma, zeta, eta, a)
#psi_hat_max = np.sqrt(np.max(np.sum(np.sum(np.abs(psi_hat)**2, 3), 2)))
#psi_hat = psi_hat / psi_hat_max
# print(psi_hat.shape)

phi_hat = low_pass_freq(n, sigma_low_pass, sigma_low_pass)

# ensure \sum ||\psi_{j,\theta}||_2^2 + ||\phi||_2^2 = 1
# wave_sum = np.sqrt(np.sum(np.abs(psi_hat)**2) + np.sum(np.abs(phi_hat)**2)) / (2 * pi)
# psi_hat = psi_hat / wave_sum
# phi_hat = phi_hat / wave_sum
# print('wavelet l2 sum: ', np.sum(np.abs(psi_hat)**2) + np.sum(np.abs(phi_hat)**2))
    
if layer2 or cov:
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
    if layer2:
        scat = scattering_2d_2layers(phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag, second_all=False)
    else:
        scat = scattering_2d_cov(phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag)
    if torch.cuda.is_available():
        print(True)
        scat = scat.cuda()
else:
    psi_hat = np.reshape(psi_hat, (n,n,-1))
    phi_hat = np.reshape(phi_hat, (n,n,-1))
    psi_hat = np.concatenate((phi_hat, psi_hat),2)
    psi_hat_real = torch.from_numpy(np.real(np.fft.fftshift(psi_hat, axes = (0,1)))).float()
    psi_hat_imag = torch.from_numpy(np.imag(np.fft.fftshift(psi_hat, axes = (0,1)))).float()
    if torch.cuda.is_available():
        psi_hat_real = psi_hat_real.cuda()
        psi_hat_imag = psi_hat_imag.cuda()    
    # initialize scattering module
    scat = scattering_2d(psi_hat_real, psi_hat_imag)
    if torch.cuda.is_available():
        print(True)
        scat = scat.cuda()

# load data
x = loadmat('./data/mydata.mat')
x = x['mydata']
x_data = torch.from_numpy(x).float()
m = int((x.shape[0] - n)/2)
target = x_data[m:(m+n), m:(m+n), :]
del x, x_data 
# id_all = [2,3,5,8,10,11,13,14,15,18,19,20,22,30,31,32,34,35,36]
id_all = [2,3,5,10,13,14,15,18,19,20,22,32,34,35,36]

f=open("result.txt", "a+")
f.write("test: %d,\n image type: textures, \n image id: %s, \n image size: %d,\n K: %d, J: %f, Q: %d \n sigma of low pass: %f, \n learning rate: %f,\n iteration interval to save: %d,\n iteration interval to append error: %d,\n reduced error by: %10.8f,\n initial type:%s ,\n 2nd layer: %r, \n covariance: %r. \n \n "%(test_id, str(id_all), n, K, J, Q, sigma_low_pass, lr, nit, err_it, min_error, initial_type, layer2, cov))
f.close()
for image_id in id_all:    
    print('image: ', image_id)
    target_temp = target[:,:,image_id] #/ torch.sqrt(torch.sum(target[:,:,image_id]**2))
    synthesis(target_temp, test_id, image_id, scat, n, min_error, err_it, nit, initial_type = initial_type)

