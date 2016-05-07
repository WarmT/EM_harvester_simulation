from __future__ import division
import numpy as np
from numpy import cos, power, sqrt, pi, power, multiply, array
from scipy.interpolate import interp1d
from numba import jit

@jit
def calc_force_dist(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, h, Nr, Nphi):
#    Nr = 20 
#    Nphi = 60
    mu0 = 1.256637e-6
    m1_Ms = m1_Br/mu0          # 1.32 T / mu0
    m2_Ms = m2_Br/mu0          # 1.32 T / mu0
    m1_Rslice = (m1_r*m1_r)/Nr
    m2_Rslice = (m2_r*m2_r)/Nr
    const = mu0*m1_Ms*m2_Ms*Nphi/(4*pi)*multiply(pi*m1_Rslice/Nphi, pi*m2_Rslice/Nphi)

    delta_phi = 2*pi/Nphi
    hm = array([h, -(h+m1_t), -(h+m2_t), (h+m1_t+m2_t)], dtype='float')
    S1 = 0.0
    R_i_prev = 0.0
    for i in range(0, Nr):
        R_i = sqrt(R_i_prev*R_i_prev + m1_Rslice)
        if i == 0:
            ri = 2*R_i*np.sin(delta_phi/2)/(delta_phi/2*3)
        else:
            ri = 2*(R_i*R_i + R_i*R_i_prev + R_i_prev*R_i_prev)*np.sin(delta_phi/2)/(delta_phi/2*3*(R_i+R_i_prev))
#            ri = (R_i+R_i_prev)/2
#        ri = (R_i+R_i_prev)/2
        R_i_prev = R_i
        S2 = 0.0
        R_ii_prev = 0.0
        for ii in range(0, Nr):
            R_ii = sqrt(R_ii_prev*R_ii_prev + m2_Rslice)
            if ii == 0:
                rii = 2*R_ii*np.sin(delta_phi/2)/(delta_phi/2*3)
            else:
                rii = 2*(R_ii*R_ii + R_ii*R_ii_prev + R_ii_prev*R_ii_prev)*np.sin(delta_phi/2)/(delta_phi/2*3*(R_ii+R_ii_prev))
#                rii = (R_ii+R_ii_prev)/2
#            rii = (R_ii+R_ii_prev)/2
            R_ii_prev = R_ii
            S3 = 0.0
            ri_rii_producs = ri*ri+rii*rii
            hm0_p1 = ri_rii_producs + hm[0]*hm[0]
            hm1_p1 = ri_rii_producs + hm[1]*hm[1]
            hm2_p1 = ri_rii_producs + hm[2]*hm[2]
            hm3_p1 = ri_rii_producs + hm[3]*hm[3]
            p2 = 2*ri*rii
            for j in range(0, Nphi):
                phi = j*delta_phi+delta_phi/2
                S3 += hm[0]/power(hm0_p1-p2*cos(phi), 3/2) \
                    + hm[1]/power(hm1_p1-p2*cos(phi), 3/2) \
                    + hm[2]/power(hm2_p1-p2*cos(phi), 3/2) \
                    + hm[3]/power(hm3_p1-p2*cos(phi), 3/2)
            S2 += S3
        S1 += S2
    F = const * S1
    return F

def calc_force(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, h_range, Nr, Nphi):
    F = np.zeros(h_range.size)
    for index, h in enumerate(h_range):
        F[index] = calc_force_dist(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, h, Nr, Nphi)
    return F

def calc_force_to_moving_magnet(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, Nr, Nphi, d0, h_steps):
    h_min = 0.001 # minimum distance between opposing magnets is 1 mm
    h_range = np.linspace(h_min, 2*d0, 2*h_steps)
    F = calc_force(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, h_range, Nr, Nphi)
    fem_int = interp1d(h_range, F, kind='cubic')
    
    x_int = np.linspace(-d0+2*h_min, d0-2*h_min, h_steps)
    F_int = fem_int(d0-x_int)-fem_int(d0+x_int)
    return (x_int, F_int)
