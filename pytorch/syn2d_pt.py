from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
#import pylab as plt
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
parser.add_argument('--image_id', type=str, default='2 3 5 8 10 11 13 14 15 18 19 20 22 30 31 32 34 35 36', help='id of the texture that wanted to synthesize')
parser.add_argument('--K', type=int, default=8, help='number of angles for defining wavelets')
parser.add_argument('--K2', type=int, default=6, help='number of angles for defining 2nd layer wavelets')
parser.add_argument('--J2', type=float, default=4, help='number of angles for defining 2nd layer wavelets')
parser.add_argument('--Q2', type=int, default=1, help='number of angles for defining 2nd layer wavelets')
parser.add_argument('--sigma2', type=float, default=0.9, help='number of angles for defining 2nd layer wavelets')
parser.add_argument('--zeta2', type=float, default=1.8, help='number of angles for defining 2nd layer wavelets')
parser.add_argument('--Q', type=int, default=2, help='scale intervals for defining wavelets')
parser.add_argument('--J', type=float, default=5, help='largest scale for defining wavelets')
parser.add_argument('--sigma', type=float, default=1.1, help='variance of mother wavelet')
parser.add_argument('--sigma_low_pass', type=float, default=15, help='variance of low pass')
parser.add_argument('--sigma_low_pass2', type=float, default=15, help='variance of low pass')
parser.add_argument('--zeta', type=float, default=1.2, help='bias over y direction of mother wavelet')
parser.add_argument('--eta', type=float, default=0.75 * pi, help='central frequency of mother wavelet')
parser.add_argument('--a', type=int, default=2, help='parameter to dilate wavelets')
parser.add_argument('--min_error', type=float, default=1e-8, help='how much error want to reduce to relative to initial error')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--nit', type=int, default=2000, help='iteration interval to save')
parser.add_argument('--max_it', type=int, default=1.5e5, help='maximum iterations to run')
parser.add_argument('--err_it', type=int, default=500, help='iteration interval to append error')
parser.add_argument('--initial_type', type=str, default='uniform', help='type of initial noise')
parser.add_argument('--layer2', type=int, default=0, help='whether to do 2nd layer scattering')
parser.add_argument('--cov', type=int, default=0, help='whether to compute covariance')
parser.add_argument('--partial', type=int, default=0, help='whether to compute partial covariance')
opt = parser.parse_args()
 
# read parameters
test_id=opt.test_id
n = opt.n
id_all = opt.image_id
id_all = [int(i) for i in id_all.split(' ')]
Q = opt.Q
J = opt.J
K = opt.K
K2 = opt.K2
J2 =  opt.J2
Q2 = opt.Q2
sigma2 = opt.sigma2
zeta2 = opt.zeta2
sigma_low_pass2 = opt.sigma_low_pass2

a = opt.a
sigma = opt.sigma
sigma_low_pass = opt.sigma_low_pass
zeta = opt.zeta
eta = opt.eta
min_error = opt.min_error
max_it = opt.max_it
lr = opt.lr
nit = opt.nit
err_it = opt.err_it
initial_type = opt.initial_type
layer2 = bool(opt.layer2)
partial = bool(opt.partial)
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
    def __init__(self, psi_hat_real, psi_hat_imag):
        super(scattering_2d_cov, self).__init__()
        self.psi_hat_real = psi_hat_real
        self.psi_hat_imag = psi_hat_imag
        print('scattering type: 1 layer covariance')

    def forward(self, x_hat):
        conv_freq_real = x_hat[:,:,0] * self.psi_hat_real - x_hat[:,:,1] * self.psi_hat_imag 
        # nw * n * n
        conv_freq_imag = x_hat[:,:,1] * self.psi_hat_real + x_hat[:,:,0] * self.psi_hat_imag
        conv_freq = torch.cat((conv_freq_real.unsqueeze(3), conv_freq_imag.unsqueeze(3)), 3)
        # conv_freq: nw * n * n * 2
        conv_space = torch.ifft(conv_freq, 2) # nw * n * n * 2
        conv_space_modulus = torch.sqrt(conv_space[:,:,:,0]**2 + conv_space[:,:,:,1]**2 + 1e-8) # nw * n * n
        
        nw = self.psi_hat_real.size()[0]
        for i in range(nw - 1):
            temp = torch.mean(torch.mean(conv_space_modulus[i:i+1] * conv_space_modulus[i:], 2), 1)
            if i == 0:
                s = temp
            else:
                s = torch.cat((s, temp), 0)
        return s

class scattering_2d_2layer_partial_cov(torch.nn.Module):
    def __init__(self, psi_hat_real, psi_hat_imag, psi_hat_real2, psi_hat_imag2,K,K2,Q):
        super(scattering_2d_2layer_partial_cov, self).__init__()
        self.K = K
        self.K2 = K2
        self.Q = Q
        self.n = psi_hat_real.shape[1]
        self.psi_hat_real = psi_hat_real
        self.psi_hat_imag = psi_hat_imag
        self.psi_hat2 = psi_hat_real2**2 + psi_hat_imag2**2 # low pass at the end
        print('scattering type: 2 layer partial covariance')

    def forward(self, x_hat):
        conv_freq_real = x_hat[:,:,0] * self.psi_hat_real - x_hat[:,:,1] * self.psi_hat_imag 
        # nw * n * n
        conv_freq_imag = x_hat[:,:,1] * self.psi_hat_real + x_hat[:,:,0] * self.psi_hat_imag
        conv_freq = torch.cat((conv_freq_real.unsqueeze(3), conv_freq_imag.unsqueeze(3)), 3)
        # conv_freq: nw * n * n * 2
        conv_space = torch.ifft(conv_freq, 2) # nw * n * n * 2
        conv_space_modulus = torch.sqrt(conv_space[:,:,:,0]**2 + conv_space[:,:,:,1]**2 + 1e-8) # nw * n * n
        a = conv_space_modulus[-1,:,:]
        a_hat = torch.rfft(a, 2, onesided = False)
        p = conv_space_modulus[:-1,:,:].reshape(self.K,-1,self.n,self.n) # K * J * n * n
        p_hat = torch.rfft(p, 2, onesided = False) # K * J * n * n * 2
        s = torch.sqrt(torch.mean(torch.mean((a_hat[:,:,0]**2 + a_hat[:,:,1]**2)* self.psi_hat2[-1],0),0)**2 + 1e-8).unsqueeze(0)
        
        
        for j in range(p.shape[1]):
            temp3 = self.psi_hat2[min(int(j/self.Q), 4)*K2:].unsqueeze(1)
            for i in range(self.K):
                temp1 = (p_hat[i:(i+1),j,:,:,0] * p_hat[i:,j,:,:,0] + p_hat[i:(i+1),j,:,:,1] * p_hat[i:,j,:,:,1]).unsqueeze(0)
                temp2 = (p_hat[i:(i+1),j,:,:,1] * p_hat[i:,j,:,:,0] - p_hat[i:(i+1),j,:,:,0] * p_hat[i:,j,:,:,1]).unsqueeze(0)
                temp = torch.sqrt(torch.mean(torch.mean(temp1 * temp3, 3), 2)**2 + torch.mean(torch.mean(temp2 * temp3, 3), 2)**2 + 1e-8).flatten()
                s = torch.cat((s, temp), 0)
                for l in range(j+1, p.shape[1]):
                    temp1 = (p_hat[i,j,:,:,0] * p_hat[i,l,:,:,0] + p_hat[i,j,:,:,1] * p_hat[i,l,:,:,1]).unsqueeze(0)
                    temp2 = (p_hat[i,j,:,:,1] * p_hat[i,l,:,:,0] - p_hat[i,j,:,:,0] * p_hat[i,l,:,:,1]).unsqueeze(0)
                    temp = torch.sqrt(torch.mean(torch.mean(temp1 * self.psi_hat2[min(int(l/2), 4)*K2:], 2), 1)**2 + torch.mean(torch.mean(temp2 * self.psi_hat2[min(int(l/2), 4)*K2:], 2), 1)**2 + 1e-8).flatten()
                    s = torch.cat((s, temp), 0)
                temp1 = (p_hat[i,j,:,:,0] * a_hat[:,:,0] + p_hat[i,j,:,:,1] * a_hat[:,:,1]).unsqueeze(0)
                temp2 = (p_hat[i,j,:,:,1] * a_hat[:,:,0] - p_hat[i,j,:,:,0] * a_hat[:,:,1]).unsqueeze(0)
                temp = torch.sqrt(torch.mean(torch.mean(temp1 * self.psi_hat2[4*K2:], 2), 1)**2 + torch.mean(torch.mean(temp2 * self.psi_hat2[4*K2:], 2), 1)**2 + 1e-8).flatten()
                s = torch.cat((s, temp), 0)
        return s
        
# class scattering_2d_2layer_partial_cov(torch.nn.Module):
#     def __init__(self, psi_hat_real, psi_hat_imag, K):
#         super(scattering_2d_2layer_partial_cov, self).__init__()
#         self.K = K
#         self.n = psi_hat_real.shape[1]
#         self.psi_hat_real = psi_hat_real
#         self.psi_hat_imag = psi_hat_imag
#         self.psi_hat2 = psi_hat_real**2 + psi_hat_imag**2 # low pass at the end
#         print('scattering type: 1 layer covariance')

#     def forward(self, x_hat):
#         conv_freq_real = x_hat[:,:,0] * self.psi_hat_real - x_hat[:,:,1] * self.psi_hat_imag 
#         # nw * n * n
#         conv_freq_imag = x_hat[:,:,1] * self.psi_hat_real + x_hat[:,:,0] * self.psi_hat_imag
#         conv_freq = torch.cat((conv_freq_real.unsqueeze(3), conv_freq_imag.unsqueeze(3)), 3)
#         # conv_freq: nw * n * n * 2
#         conv_space = torch.ifft(conv_freq, 2) # nw * n * n * 2
#         conv_space_modulus = torch.sqrt(conv_space[:,:,:,0]**2 + conv_space[:,:,:,1]**2 + 1e-8) # nw * n * n
#         a = conv_space_modulus[0,:,:]
#         a_hat = torch.rfft(a, 2, onesided = False)
#         p = conv_space_modulus[1:,:,:].reshape(self.K,-1,self.n,self.n) # K * J * n * n
#         p_hat = torch.rfft(p, 2, onesided = False) # K * J * n * n * 2
#         s = torch.mean(torch.mean(a**2,0),0).unsqueeze(0)
        
#         for i in range(self.K):
#             for j in range(p.shape[1]):
#                 for i1 in range(i,self.K):
#                     temp_real = torch.mean(torch.mean((p_hat[i,j,:,:,0] * p_hat[i1,j,:,:,0] + p_hat[i,j,:,:,1] * p_hat[i1,j,:,:,1]) * self.psi_hat2[(j+1)*K:], 2),1)
#                     temp_imag = torch.mean(torch.mean((p_hat[i,j,:,:,1] * p_hat[i1,j,:,:,0] - p_hat[i,j,:,:,0] * p_hat[i1,j,:,:,1]) * self.psi_hat2[(j+1)*K:], 2),1)
#                     temp = temp_real**2 + temp_imag**2
#                     s  = torch.cat((s, temp),0)
#                 for j1 in range(j+1, p.shape[1]):
#                     temp_real = torch.mean(torch.mean((p_hat[i,j,:,:,0] * p_hat[i,j1,:,:,0] + p_hat[i,j,:,:,1] * p_hat[i,j1,:,:,1]) * self.psi_hat2[(j1+1)*K:], 2),1)
#                     temp_imag = torch.mean(torch.mean((p_hat[i,j,:,:,1] * p_hat[i,j1,:,:,0] - p_hat[i,j,:,:,0] * p_hat[i,j1,:,:,1]) * self.psi_hat2[(j1+1)*K:], 2),1)
#                     temp = temp_real**2 + temp_imag**2
#                     s  = torch.cat((s, temp),0)
#                 temp_real = torch.mean(torch.mean((p_hat[i,j,:,:,0] * a_hat[:,:,0] + p_hat[i,j,:,:,1] * a_hat[:,:,1]) * self.psi_hat2[-1:], 2),1)
#                 temp_imag = torch.mean(torch.mean((p_hat[i,j,:,:,1] * a_hat[:,:,0] - p_hat[i,j,:,:,0] * a_hat[:,:,1]) * self.psi_hat2[-1:], 2),1)
#                 temp = temp_real**2 + temp_imag**2
#                 s  = torch.cat((s, temp), 0)
#         return s
    
class scattering_2d_partial_cov(torch.nn.Module):
    def __init__(self, psi_hat_real, psi_hat_imag, K):
        super(scattering_2d_partial_cov, self).__init__()
        self.psi_hat_real = psi_hat_real
        self.psi_hat_imag = psi_hat_imag
        self.K = K
        self.n = psi_hat_real.shape[1]
        print('scattering type: 1 layer covariance')

    def forward(self, x_hat):
        conv_freq_real = x_hat[:,:,0] * self.psi_hat_real - x_hat[:,:,1] * self.psi_hat_imag 
        # nw * n * n
        conv_freq_imag = x_hat[:,:,1] * self.psi_hat_real + x_hat[:,:,0] * self.psi_hat_imag
        conv_freq = torch.cat((conv_freq_real.unsqueeze(3), conv_freq_imag.unsqueeze(3)), 3)
        # conv_freq: nw * n * n * 2
        conv_space = torch.ifft(conv_freq, 2) # nw * n * n * 2
        conv_space_modulus = torch.sqrt(conv_space[:,:,:,0]**2 + conv_space[:,:,:,1]**2 + 1e-8) # nw * n * n
        a = conv_space_modulus[0,:,:]
        p = conv_space_modulus[1:,:,:].reshape(self.K,-1,self.n,self.n)
        s = torch.mean(torch.mean(a**2,0),0).unsqueeze(0)
        
        for i in range(self.K):
            for j in range(p.shape[1]):
                temp = torch.mean(torch.mean(torch.cat((p[i,j:(j+1)]**2, p[i,j:(j+1)]*a),0), 2), 1)
                if i < self.K - 1 and j < p.shape[1]-1:
                    temp1 = torch.mean(torch.mean(p[i,j] * p[i+1:,j], 2), 1)
                    temp2 = torch.mean(torch.mean(p[i,j] * p[i,j+1:], 2), 1)
                    s = torch.cat((s, temp, temp1, temp2), 0)
                elif i < self.K - 1:
                    temp1 = torch.mean(torch.mean(p[i,j] * p[i+1:,j], 2), 1)
                    s = torch.cat((s, temp, temp1), 0)
                elif j < p.shape[1]  - 1:
                    temp2 = torch.mean(torch.mean(p[i,j] * p[i,j+1:], 2), 1)
                    s = torch.cat((s, temp, temp2), 0)
                else:
                    s = torch.cat((s, temp), 0)
                    
        return s
    
   
class scattering_2d_2layers(torch.nn.Module):
    def __init__(self, phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag, K, second_all = False):
        super(scattering_2d_2layers, self).__init__()
        self.phi_hat_real = phi_hat_real # n * n
        self.phi_hat_imag = phi_hat_imag # n * n
        self.psi_hat_real = psi_hat_real # K * J * n * n
        self.psi_hat_imag = psi_hat_imag # K * J * n * n
        self.second_all = second_all # whether to do all pair of (j1, j2) at second layer or just j2 > j1
        self.K = K
        print('scattering type: 2 layers.')
        
    def forward(self, x_hat):
        # x_hat: n * n * 2
        s = 1/(2 * pi)**2 * torch.mean(torch.mean((x_hat[:,:,0]**2 + x_hat[:,:,1]**2) * (self.phi_hat_real**2 + self.phi_hat_imag**2), 0), 0)
        s = s.unsqueeze(0)
        J = self.psi_hat_real.size()[1]
        for i in range(J):
            temp_real = x_hat[:,:,0] * self.psi_hat_real[:self.K,i,:,:] - x_hat[:,:,1] * self.psi_hat_imag[:self.K,i,:,:] # first layer only use gabor wavelets which has K angles
            temp_imag = x_hat[:,:,0] * self.psi_hat_imag[:self.K,i,:,:] + x_hat[:,:,1] * self.psi_hat_real[:self.K,i,:,:]
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

def Dealias(u, n):
    uk = np.fft.fftshift(np.fft.fft2(u))
    x = np.linspace(-pi,pi - 2*pi/n,n)
    y = np.linspace(-pi,pi - 2*pi/n,n)
    x, y = np.meshgrid(x,y)
    K = x**2 + y**2
    kcut = (1/3) * np.max(K)
    ind = np.where(kcut < K)
    uk[ind] = 0
    u = np.fft.ifft2(np.fft.fftshift(uk))
    return np.real(u)

def synthesis(target, test_id, ind, scat, n, min_error, max_it, err_it, nit, is_complex = False, initial_type = 'gaussian'):
    if torch.cuda.is_available():
        target = target.cuda()
    # set up target
#     target_ = n**2 * target / torch.sqrt(torch.sum(target**2))

    target_hat = torch.rfft(target, 2, onesided = False)
    s_target = scat(target_hat)
    print('shape of s:', s_target.size())
#    s_target = s_target / torch.sum(s_target) 

    if initial_type == 'gaussian':
        x0 = np.random.randn(n,n)
    elif initial_type == 'uniform':
        x0 = np.random.rand(n,n)
#     x0 = Dealias(x0,n)
    x0 = torch.from_numpy(x0).float()
    if torch.cuda.is_available():
        s_target = s_target.cuda()
        x0 = x0.cuda()
#     x0 = n**2 * x0/torch.sqrt(torch.sum(target**2))
    x0 = Variable(x0, requires_grad=True)
    
    x0_hat = torch.rfft(x0, 2, onesided = False)
    
    s0 = scat(x0_hat)
    loss = nn.MSELoss()
    optimizer = optim.Adam([x0], lr=lr)
    output = loss(s_target, s0)
    l0 = output
    error = []
    count = 0
    while (output / l0 > min_error) and (count < max_it):
        optimizer.zero_grad() 
        x0_hat = torch.rfft(x0, 2, onesided = False)
        s0 = scat(x0_hat)
        output = loss(s_target, s0)
        if count % err_it ==0:
        #if count % 1 == 0:   
            error.append(output.item())
        output.backward()
        optimizer.step()
        output = loss(s_target, s0)
        if count % nit == 0:
       # if count % 1 == 0:
            print(output.data.cpu().numpy())
            if not math.isnan(output):
                np.save('./result%d/syn_error%d.npy'%(test_id, ind), np.asarray(error))
                np.save('./result%d/syn_result%d.npy'%(test_id, ind), x0.data.cpu().numpy())
            # plot_image(x0.data.cpu().numpy(), test_id, ind, count, nit)
        count += 1 
    print('error reduced by: ', output / l0)
    print('error supposed reduced by: ', min_error)
    if not math.isnan(output):
        np.save('./result%d/syn_error%d.npy'%(test_id, ind), np.asarray(error))
        np.save('./result%d/syn_result%d.npy'%(test_id, ind), x0.data.cpu().numpy())

# define wavelets
psi_hat = gabor_wavelet_family_freq_2d(n, K, J, Q, sigma, zeta, eta, a)
phi_hat = low_pass_freq(n, sigma_low_pass, sigma_low_pass)
psi_hat2 = gabor_wavelet_family_freq_2d(n, K2, J2, Q2, sigma2, zeta2, eta, a)
psi_hat2 = psi_hat2[:,:,:,1:]
phi_hat2 = low_pass_freq(n, sigma_low_pass2, sigma_low_pass2)

if (layer2 and cov) or (not layer2 and not cov):
    psi_hat, phi_hat = np.reshape(psi_hat, (n,n,-1)), np.reshape(phi_hat, (n,n,-1))
    psi_hat = np.concatenate((psi_hat, phi_hat),2)
    psi_hat2, phi_hat2 = np.reshape(psi_hat2, (n,n,-1)), np.reshape(phi_hat2, (n,n,-1))
    psi_hat2 = np.concatenate((psi_hat2, phi_hat2),2)

    psi_hat = np.swapaxes(psi_hat, 0,2)
    psi_hat = np.swapaxes(psi_hat, 1,2)
    psi_hat2 = np.swapaxes(psi_hat2, 0,2)
    psi_hat2 = np.swapaxes(psi_hat2, 1,2)
    psi_hat_real = torch.from_numpy(np.real(np.fft.fftshift(psi_hat, axes = (1,2)))).float()
    psi_hat_imag = torch.from_numpy(np.imag(np.fft.fftshift(psi_hat, axes = (1,2)))).float()
    psi_hat_real2 = torch.from_numpy(np.real(np.fft.fftshift(psi_hat2, axes = (1,2)))).float()
    psi_hat_imag2 = torch.from_numpy(np.imag(np.fft.fftshift(psi_hat2, axes = (1,2)))).float()
    if torch.cuda.is_available():
        psi_hat_real = psi_hat_real.cuda()
        psi_hat_imag = psi_hat_imag.cuda() 
        psi_hat_real2 = psi_hat_real2.cuda()
        psi_hat_imag2 = psi_hat_imag2.cuda() 
    if layer2:
        scat = scattering_2d_2layer_partial_cov(psi_hat_real, psi_hat_imag, psi_hat_real2, psi_hat_imag2, K, K2,Q)
    else:
        scat = scattering_2d(psi_hat_real, psi_hat_imag)
elif layer2:
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
    scat = scattering_2d_2layers(phi_hat_real, phi_hat_imag, psi_hat_real, psi_hat_imag, K, second_all=False)
elif cov:
    psi_hat, phi_hat = np.reshape(psi_hat, (n,n,-1)), np.reshape(phi_hat, (n,n,-1))
    psi_hat = np.concatenate((phi_hat, psi_hat),2)
    psi_hat = np.swapaxes(psi_hat, 0,2)
    psi_hat = np.swapaxes(psi_hat, 1,2)
    psi_hat_real = torch.from_numpy(np.real(np.fft.fftshift(psi_hat, axes = (1,2)))).float()
    psi_hat_imag = torch.from_numpy(np.imag(np.fft.fftshift(psi_hat, axes = (1,2)))).float()
    if torch.cuda.is_available():
        psi_hat_real = psi_hat_real.cuda()
        psi_hat_imag = psi_hat_imag.cuda()    
    if not partial:
        scat = scattering_2d_cov(psi_hat_real, psi_hat_imag)
    else:
        scat = scattering_2d_partial_cov(psi_hat_real, psi_hat_imag,K)
    
    
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
##id_all = [3,8,11,13,30,36]
#id_all = [2,3,5,8,10,11,13,14,15,18,19,20,22,30,32,34,35,36]
f=open("result.txt", "a+")
f.write("test: %d,\n image type: textures, \n image id: %s, \n image size: %d,\n K: %d, J: %f, Q: %d,  sigma: %f, \zeta: %f, eta: %f, \n sigma of low pass: %f, \n learning rate: %f,\n iteration interval to save: %d,\n maximum iterations to run: %d,\n iteration interval to append error: %d,\n reduced error by: %10.8f,\n initial type:%s ,\n 2nd layer: %r, \n covariance: %r, \n partial covariance: %r, \n "%(test_id, str(id_all), n, K, J, Q, sigma, zeta, eta, sigma_low_pass, lr, nit, max_it, err_it, min_error, initial_type, layer2, cov, partial))
f.close()
for image_id in id_all:    
    print('image: ', image_id)
#     target_temp = torch.from_numpy(Dealias(target[:,:,image_id].data.cpu().numpy(),n)).float() # dealias the fft of textures  #/ torch.sqrt(torch.sum(target[:,:,image_id]**2))
    target_temp = target[:,:,image_id]
    synthesis(target_temp, test_id, image_id, scat, n, min_error, max_it, err_it, nit, initial_type = initial_type)

