from __future__ import division
import numpy as np
from numpy import cos, power, sqrt, pi, multiply, array
from scipy.interpolate import interp1d
from numba import jit


@jit
def calc_force_dist(r_m1, h_m1, Br_m1, r_m2, h_m2, Br_m2, d, Nr, Nphi):
    mu0 = 1.256637e-6
    m1_Ms = Br_m1 / mu0
    m2_Ms = Br_m2 / mu0
    m1_Rslice = (r_m1 * r_m1) / Nr
    m2_Rslice = (r_m2 * r_m2) / Nr
    const = mu0 * m1_Ms * m2_Ms * Nphi / (4 * pi) * multiply(pi * m1_Rslice / Nphi, pi * m2_Rslice / Nphi)

    delta_phi = 2 * pi / Nphi
    hm = array([d, -(d + h_m1), -(d + h_m2), (d + h_m1 + h_m2)], dtype='float')
    S1 = 0.0
    R_i_prev = 0.0
    for i in range(0, Nr):
        R_i = sqrt(R_i_prev * R_i_prev + m1_Rslice)
        ri = (R_i + R_i_prev) / 2
        R_i_prev = R_i
        S2 = 0.0
        R_ii_prev = 0.0
        for ii in range(0, Nr):
            R_ii = sqrt(R_ii_prev * R_ii_prev + m2_Rslice)
            rii = (R_ii + R_ii_prev) / 2
            R_ii_prev = R_ii
            S3 = 0.0
            ri_rii_producs = ri * ri + rii * rii
            hm0_p1 = ri_rii_producs + hm[0] * hm[0]
            hm1_p1 = ri_rii_producs + hm[1] * hm[1]
            hm2_p1 = ri_rii_producs + hm[2] * hm[2]
            hm3_p1 = ri_rii_producs + hm[3] * hm[3]
            p2 = 2 * ri * rii
            for j in range(0, Nphi):
                phi = j * delta_phi + delta_phi / 2
                S3 += hm[0] / power(hm0_p1 - p2 * cos(phi), 3 / 2) \
                    + hm[1] / power(hm1_p1 - p2 * cos(phi), 3 / 2) \
                    + hm[2] / power(hm2_p1 - p2 * cos(phi), 3 / 2) \
                    + hm[3] / power(hm3_p1 - p2 * cos(phi), 3 / 2)
            S2 += S3
        S1 += S2
    F = const * S1
    return F


def calc_force(r_m1, h_m1, Br_m1, r_m2, h_m2, Br_m2, h_range, Nr, Nphi):
    F = np.zeros(h_range.size)
    for index, d in enumerate(h_range):
        F[index] = calc_force_dist(r_m1, h_m1, Br_m1, r_m2, h_m2, Br_m2, d, Nr, Nphi)
    return F


def calc_force_to_moving_magnet(r_m1, h_m1, Br_m1, r_m2, h_m2, Br_m2, Nr, Nphi, d_sep, h_steps):
    h_min = 0.001  # minimum distance between opposing magnets is 1 mm
    h_range = np.linspace(h_min, 2 * d_sep, 2 * h_steps)
    F = calc_force(r_m1, h_m1, Br_m1, r_m2, h_m2, Br_m2, h_range, Nr, Nphi)
    fem_int = interp1d(h_range, F, kind='cubic')

    x_int = np.linspace(-d_sep + 2 * h_min, d_sep - 2 * h_min, h_steps)
    F_int = fem_int(d_sep - x_int) - fem_int(d_sep + x_int)
    return (x_int, F_int)
