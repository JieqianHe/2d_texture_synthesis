# 2d_texture_synthesis

Synthesis 2d texture by matching l1 or l2 scattering norms. 

synthesis_2d_l2.py: initially start with 4 wavelets, then iteratively insert more wavelets. Computation is done in frequency field, with the constrains that x_hat(-omega) = \bar(x_hat(omega)).

syn2d_pt.py: synthesis texture images in pytorch with choice of 1 layer variance, 1 layer covariance and 2 layers. Do not need to specify gradients. 

syn2d_pt_nvstks.py: synthesis on navier stokes. Difference from textures: complex valued input, need torch.fft instead of torch.rfft, need wavelet with angles [0, 2\pi) instead of [0, \pi).  

