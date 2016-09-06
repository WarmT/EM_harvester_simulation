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


def plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, print_P, d_co, N, norm):
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

    print "d_co = %d um, c_r = %.1f, c_h = %.1f, P_max = %.2f mW, V(P_max) = %.2f V, Rc = %d, Rl = %d, Z = %.2f, N = %d" % \
        (int(d_co * 1e6), P_xx, P_yy, P_max, V[P_max_y][P_max_x], int(round(Rc[P_max_y][P_max_x])),
         int(round(Rl[P_max_y][P_max_x])), Z[P_max_y][P_max_x], int(round(N[P_max_y][P_max_x])))

    if norm:
        P /= P_max

    fig = plt.figure(facecolor='white', figsize=(17, 13))

    legend_location = 3
    mark_size = 8
    contour_lines = 10
    plt.subplot(3, 2, 1)
    CS = plt.contour(X, Y, Z, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title("Max displacement Z (mm)")

    plt.subplot(3, 2, 2)
    CS = plt.contour(X, Y, Rc, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title(r"Coil resistance ($\Omega$)")

    plt.subplot(3, 2, 3)
    CS = plt.contour(X, Y, Rl, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title(r"Load resistance ($\Omega$)")

    plt.subplot(3, 2, 4)
    CS = plt.contour(X, Y, k, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%0.1f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title("magnetic flux gradient (V/(m/s))")

    plt.subplot(3, 2, 5)
    CS = plt.contour(X, Y, V, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%0.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title("Load voltage (V)")

    plt.subplot(3, 2, 6)
#    CS = plt.contour(X, Y, P, 20, cmap=plt.get_cmap('viridis_r'))
    CS = plt.contour(X, Y, P, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%0.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)

    if print_P:
        ax = plt.gca()
        text_offset = 2.0
        ax.text(P_xx, P_yy + text_offset, "%.2f mW" % (P_max), fontsize=15)
#        ax.text(P_xx+text_offset, P_yy-2*text_offset, "(%.2f, %.2f)" % (P_xx, P_yy), fontsize=15)
    plt.title(r"Output power (mW)")
    plt.ylabel(r"$Coil_\mathrm{h}$")
    plt.xlabel(r"$Coil_\mathrm{r1}$")

    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig(outfile)
    raw_input("tadaa!")
    plt.close()


def draw_all_contours(outfile, m_Br, h, coil_r2, gap, t0_per_h_coil, d_co, k_co, a, f, two_coils, norm):

    step = 0.2 / 1000
    min_r = 1.0 / 1000
    min_h = d_co
    x = np.arange(min_r, coil_r2, step)
    y = np.arange(min_h, h, step)
    X, Y = np.meshgrid(x * 1000, y * 1000)

    Z  = np.zeros_like(X)
    Rc = np.zeros_like(X)
    Rl = np.zeros_like(X)
    k  = np.zeros_like(X)
    V  = np.zeros_like(X)
    P  = np.zeros_like(X)
    N_arr  = np.zeros_like(X)

#    start = time.time()
    for i, coil_r1 in enumerate(x):
        for j, coil_h in enumerate(y):
            m_r = coil_r1 - gap
            t0 = t0_per_h_coil * coil_h
            N = int(round(4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co * d_co * np.pi)))
            N_arr[j, i] = N
            if two_coils:
                m_h = h - 2 * coil_h + 2 * t0
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
            else:
                m_h = h - coil_h + t0
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    Z *= 1000   # convert displacement to mm
    P *= 1000   # convert power to mW

#    end = time.time()
#    print "Elapsed time : %.2f seconds to calculate %d points (%.1f/s)" % (end-start, x.size*y.size, x.size*y.size/(end-start))  

    plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, False, d_co, N_arr, norm)


def main():

    m_Br = 1.1
    coil_r2 = 6.0 / 1000
    gap = 0.5 / 1000
    d_co = 40e-6
    k_co = 0.6

#    t0 = t0_per_h * coil_h
    t0_per_h_coil = 0.75

#    h = 0.01**3/(np.pi*coil_r2*coil_r2)
    h = 8.9 / 1000
#    h = 9.0 / 1000
    a = 10.0
    f = 100.0
    draw_all_contours("Spreemann_one_coil.pdf", m_Br, h, coil_r2, gap, t0_per_h_coil, d_co, k_co, a, f, False, False)
    draw_all_contours("Spreemann_two_coils.pdf", m_Br, h, coil_r2, gap, t0_per_h_coil, d_co, k_co, a, f, True, False)


if __name__ == "__main__":
    main()
