from __future__ import division
import numpy as np
from numba import jit

import matplotlib.pyplot as plt

from magnet_flux import *

import time


def plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, print_P, d_co, N, norm, h_mag, r_i, t0_per_h_coil, two_coils, ask):
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

    V_load = V[P_max_y][P_max_x]
    r_o = P_xx
    h_coil = P_yy
    if two_coils:
        coils = "Two coils:"
    else:
        coils = "One coil: "

    r_i = r_i * 1000  # convert to mm
    print "%s P_load = %.2f mW, V_load = %.2f V, r_i = %.2f mm, r_o = %.2f mm, h_coil = %.2f mm, t0/h_coil = %.3f" % (coils, P_max, V_load, r_i, r_o, h_coil, t0_per_h_coil)

    if norm:
        P /= P_max

    plt.figure(facecolor='white', figsize=(12, 12))

    legend_location = 1
    mark_size = 8
    contour_lines = 12
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
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$", fontsize='x-large')
#    plt.legend(loc=legend_location)
    plt.title("magnetic flux gradient (V/(m/s))")

    plt.subplot(2, 3, 5)
    CS = plt.contour(X, Y, V, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt='%0.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
#    plt.ylabel(r"$h_\mathrm{coil}\,\,[\mathrm{mm}]$", fontsize='x-large')
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$", fontsize='x-large')
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
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$", fontsize='x-large')
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


def draw_all_contours(outfile, m_Br, h_mag, r_mag, r_o_max, h_coil_max, gap, t0_per_h_coil, d_co, k_co, a, f, two_coils, norm, ask):

    step = 0.05 / 1000
    min_r = r_mag + gap + d_co
    min_h = d_co
    x = np.arange(min_r, r_o_max, step)
    y = np.arange(min_h, h_coil_max, step)
    X, Y = np.meshgrid(x * 1000, y * 1000)

    Z  = np.zeros_like(X)
    Rc = np.zeros_like(X)
    Rl = np.zeros_like(X)
    k  = np.zeros_like(X)
    V  = np.zeros_like(X)
    P  = np.zeros_like(X)
    N_arr  = np.zeros_like(X)

    r_i = r_mag + gap

    start = time.time()
    for i, r_o in enumerate(x):
        for j, h_coil in enumerate(y):
            t0 = t0_per_h_coil * h_coil
            N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
            N_arr[j, i] = N
            if two_coils:
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
            else:
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
    Z *= 1000   # convert displacement to mm
    P *= 1000   # convert power to mW

    end = time.time()
    print "Elapsed time : %.2f seconds to calculate %d points (%.1f/s)" % (end - start, x.size * y.size, x.size * y.size / (end - start))

    print_P = True
    plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, print_P, d_co, N_arr, norm, h_mag, r_i, t0_per_h_coil,  two_coils, ask)


def main():


    a = 10.0
    f = 100.0

    r_mag = 9.525e-3 / 2
    h_mag = 19.05e-3

#    gap = 0.5e-3
    gap = 1.26e-3
    m_Br = 1.31
    d_co = 150e-6
#    k_co = 0.790 * 0.907  # k_co for d_co = 100 um
    k_co = 0.812 * 0.907  # k_co for d_co = 150 um
    r_o_max = r_mag * 6
    h_coil_max = h_mag * 2

    t0_per_h_coil = 0.43
    draw_all_contours("pics/magnet_fixed_one_coil_contour.pdf", m_Br, h_mag, r_mag, r_o_max, h_coil_max, gap, t0_per_h_coil, d_co, k_co, a, f, False, False, True)
#    draw_all_contours("pics/magnet_fixed_two_coils_contour.pdf", m_Br, h_mag, r_mag, r_o_max, h_coil_max, gap, t0_per_h_coil, d_co, k_co, a, f, True, False, True)



if __name__ == "__main__":
    main()
