from __future__ import division
from scipy.special import ellipk, ellipe, ellipkinc, ellipeinc
from sympy import elliptic_k, elliptic_e, elliptic_pi
#from scipy import special
import numpy as np
from numba import jit
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from magnet_flux import *

import time


def plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, print_P, d_co, N, norm, h, t0_per_h_coil, two_coils, ask):
    P_max_y, P_max_x = np.unravel_index(np.argmax(P), P.shape)
    P_max = P[P_max_y][P_max_x]
    P_xx  = X[P_max_y][P_max_x]
    P_yy  = Y[P_max_y][P_max_x]

    k_max_y, k_max_x = np.unravel_index(np.argmax(k), k.shape)
#    k_max = k[k_max_y][k_max_x]
    k_xx  = X[k_max_y][k_max_x]
    k_yy  = Y[k_max_y][k_max_x]

    V_max_y, V_max_x = np.unravel_index(np.argmax(V), V.shape)
#    V_max = V[V_max_y][V_max_x]
    V_xx  = X[V_max_y][V_max_x]
    V_yy  = Y[V_max_y][V_max_x]

    h = h * 1000  # convert to mm
    V_load = V[P_max_y][P_max_x]
    r_i = P_xx
    h_coil = P_yy
    t0 = t0_per_h_coil * h_coil
    if two_coils:
        h_mag = h - 2 * h_coil + 2 * t0
        coils = "Two coils:"
    else:
        h_mag = h - h_coil + t0
        coils = "One coil: "

    h_mag_per_h = h_mag / h

    print "%s P_load = %.2f mW, V_load = %.2f V, r_i = %.2f mm, h_coil = %.2f mm, h_mag/h = %.3f, t0/h_coil = %.3f" % (coils, P_max, V_load, r_i, h_coil, h_mag_per_h, t0_per_h_coil)

#    print "d_co = %d um, c_r = %.1f, c_h = %.1f, P_max = %.2f mW, V(P_max) = %.2f V, Rc = %d, Rl = %d, Z = %.2f, N = %d" % \
#        (int(d_co * 1e6), P_xx, P_yy, P_max, V[P_max_y][P_max_x], int(round(Rc[P_max_y][P_max_x])),
#         int(round(Rl[P_max_y][P_max_x])), Z[P_max_y][P_max_x], int(round(N[P_max_y][P_max_x])))

    if norm:
        P /= P_max

    fig = plt.figure(facecolor='white', figsize=(12, 12))

    legend_location = 3
    mark_size = 8
    contour_lines = 10
    plt.subplot(2, 3, 1)
    CS = plt.contour(X, Y, Z, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.ylabel(r"$h_\mathrm{coil}\,\,[\mathrm{mm}]$", fontsize='x-large')
#    plt.xlabel(r"$r_\mathrm{i}$", fontsize='x-large')
#    plt.legend(loc=legend_location)
    plt.title("Max displacement Z (mm)")

    plt.subplot(2, 3, 2)
    CS = plt.contour(X, Y, Rc, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
#    plt.ylabel(r"$h_\mathrm{coil}\,\,[\mathrm{mm}]$", fontsize='x-large')
#   plt.xlabel(r"$r_\mathrm{i}$", fontsize='x-large')
#    plt.legend(loc=legend_location)
    plt.title(r"Coil resistance ($\Omega$)")

    plt.subplot(2, 3, 3)
    CS = plt.contour(X, Y, Rl, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
#    plt.ylabel(r"$h_\mathrm{coil}\,\,[\mathrm{mm}]$", fontsize='x-large')
#    plt.xlabel(r"$r_\mathrm{i}$", fontsize='x-large')
#    plt.legend(loc=legend_location)
    plt.title(r"Load resistance ($\Omega$)")

    plt.subplot(2, 3, 4)
    CS = plt.contour(X, Y, k, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%0.1f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.ylabel(r"$h_\mathrm{coil}\,\,[\mathrm{mm}]$", fontsize='x-large')
    plt.xlabel(r"$r_\mathrm{i}\,\,[\mathrm{mm}]$", fontsize='x-large')
#    plt.legend(loc=legend_location)
    plt.title("magnetic flux gradient (V/(m/s))")

    plt.subplot(2, 3, 5)
    CS = plt.contour(X, Y, V, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%0.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
#    plt.ylabel(r"$h_\mathrm{coil}\,\,[\mathrm{mm}]$", fontsize='x-large')
    plt.xlabel(r"$r_\mathrm{i}\,\,[\mathrm{mm}]$", fontsize='x-large')
#    plt.legend(loc=legend_location)
    plt.title("Load voltage (V)")

    plt.subplot(2, 3, 6)
#    CS = plt.contour(X, Y, P, 20, cmap=plt.get_cmap('viridis_r'))
    CS = plt.contour(X, Y, P, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%0.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
#    plt.ylabel(r"$h_\mathrm{coil}\,\,[\mathrm{mm}]$", fontsize='x-large')
    plt.xlabel(r"$r_\mathrm{i}\,\,[\mathrm{mm}]$", fontsize='x-large')
    plt.legend(loc=legend_location)
    plt.title(r"Output power (mW)")

    if print_P:
        ax = plt.gca()
        text_offset = 2.0
        ax.text(P_xx, P_yy + text_offset, "%.2f mW" % (P_max), fontsize=15)
#        ax.text(P_xx+text_offset, P_yy-2*text_offset, "(%.2f, %.2f)" % (P_xx, P_yy), fontsize=15)

    plt.subplots_adjust(wspace=0.14, hspace=0.14, left=0.05, right=0.99, top=0.97, bottom=0.06)
#    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig(outfile)
    if ask:
        raw_input("Hit any key!")
    plt.close()


def draw_all_contours(outfile, m_Br, h, r_o, gap, t0_per_h_coil, d_co, k_co, a, f, two_coils, norm, ask):

    step = 0.05 / 1000
    min_r = 1.0 / 1000
    min_h = d_co
    x = np.arange(min_r, r_o, step)
    y = np.arange(min_h, h, step)
    X, Y = np.meshgrid(x * 1000, y * 1000)

    Z  = np.zeros_like(X)
    Rc = np.zeros_like(X)
    Rl = np.zeros_like(X)
    k  = np.zeros_like(X)
    V  = np.zeros_like(X)
    P  = np.zeros_like(X)
    N_arr  = np.zeros_like(X)

    start = time.time()
    for i, r_i in enumerate(x):
        for j, h_coil in enumerate(y):
            m_r = r_i - gap
            t0 = t0_per_h_coil * h_coil
            N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
            N_arr[j, i] = N
            if two_coils:
                h_mag = h - 2 * h_coil + 2 * t0
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all_two_coils(m_Br, h_mag, m_r, h_coil, r_i, r_o, N, d_co, t0, a, f)
            else:
                h_mag = h - h_coil + t0
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all(m_Br, h_mag, m_r, h_coil, r_i, r_o, N, d_co, t0, a, f)
    Z *= 1000   # convert displacement to mm
    P *= 1000   # convert power to mW

    end = time.time()
    print "Elapsed time : %.2f seconds to calculate %d points (%.1f/s)" % (end-start, x.size*y.size, x.size*y.size/(end-start))  

    plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, False, d_co, N_arr, norm, h, t0_per_h_coil, two_coils, ask)


def main():

    m_Br = 1.1
    r_o = 6.0 / 1000
    gap = 0.5 / 1000
    d_co = 40e-6
    k_co = 0.6


#    h = 0.01**3/(np.pi*r_o*r_o)
    h = 8.9 / 1000
#    h = 9.0 / 1000
    a = 10.0
    f = 100.0

#    t0 = t0_per_h * h_coil
    t0_per_h_coil = 0.75
    draw_all_contours("Spreemann_one_coil.pdf", m_Br, h, r_o, gap, t0_per_h_coil, d_co, k_co, a, f, False, False, True)
#    draw_all_contours("Spreemann_two_coils.pdf", m_Br, h, r_o, gap, t0_per_h_coil, d_co, k_co, a, f, True, False, False)

    t0_per_h_coil = 0.797
    draw_all_contours("Spreemann_one_coil_t0.pdf", m_Br, h, r_o, gap, t0_per_h_coil, d_co, k_co, a, f, False, False, True)
    t0_per_h_coil = 0.990
    draw_all_contours("Spreemann_two_coils_t0.pdf", m_Br, h, r_o, gap, t0_per_h_coil, d_co, k_co, a, f, True, False, True)


if __name__ == "__main__":
    main()
