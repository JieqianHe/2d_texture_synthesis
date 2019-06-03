# 2d_texture_synthesis

Synthesis 2d texture by matching l1 or l2 scattering norms. 

synthesis_2d_l2.py: initially start with 4 wavelets, then iteratively insert more wavelets. Computation is done in frequency field, with the constrains that x_hat(-omega) = \bar(x_hat(omega)).
