from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from magnet_force import *
import time
from scipy.interpolate import interp1d

def main():
    F_meas = np.array([6.0, 8.2, 11.5, 16.9, 28.4, 40.8, 48, 51.0, 65.2, 80.3,  100, 117,  174, 221, 298, 352, 400, 490], dtype='float')
    s_meas = np.array([ 55,  50,   45,   40, 34.5,   30, 28, 27.8,   25,   23, 21.2,  20, 16.5,  15,  13,  12,  11, 9.8], dtype='float')

    m = 28.49 / 1000
    g = 9.819
    F_meas = np.multiply(F_meas, g/1000) # convert grams to Newtons
    s_meas = np.divide(s_meas, 1000) # convert mm to m

#    d_sep = 0.029225
    d_sep = 28e-3
    xs = 0.002225

    h_range = np.arange(0.002, 0.062, 0.001)

    m1_r = 0.015875/2; m1_t = 0.01905; m1_Br = 1.2
    m2_r = 0.015000/2; m2_t = 0.00300; m2_Br = 1.2
    Nr = 100
    Nphi = 90
    start = time.time()
    F0 = calc_force_dist(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, h_range, Nr, Nphi)
    m1_r = 0.015875/2; m1_t = 0.01905; m1_Br = 1.32
    m2_r = 0.015000/2; m2_t = 0.00300; m2_Br = 1.37
    F = calc_force_dist(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, h_range, Nr, Nphi)
    end = time.time()
    print "\nForce calculation takes %.4f seconds.\n" % (end-start)
    xp = np.linspace(0.002,0.062, 61)
    fig = plt.figure(facecolor='white', figsize=(16, 9))
    plt.axhline(0, 0, 60, color='black')
    plt.errorbar(np.multiply(s_meas,1000), F_meas, xerr=0.6, yerr=(2e-3 * g), label='Measurements')
    #plt.plot(np.multiply(s,1000), F, 'o', label='Measurements')
    # plt.plot(np.multiply(xp, 1000), pp3(xp), '-', label="3rd order polyfit")
    # plt.plot(np.multiply(xp, 1000), pp5(xp), '-', label="5th order polyfit")
    plt.plot(np.multiply(h_range,1000), F0, '-', lw=2, label=r"FEM, $B_\mathrm{r} = 1.2\,\mathrm{T}$")
    plt.plot(np.multiply(h_range,1000), F, '-', lw=2, label=r"FEM, $B_\mathrm{r} = 1.32/1.37\,\mathrm{T}$")
    plt.legend()
    plt.axis([0.0, 60, -2.0, 6.0])
    plt.xlabel(r"$d_\mathrm{sep}\,\,[\,\mathrm{mm}\,]$", fontsize='x-large')
    plt.ylabel('Repulsion Force [ N ]')
    ax = plt.gca()
    ax.add_patch(patches.Rectangle((20, 4), 3, 1.5, facecolor='grey', alpha=0.2))
    ax.text(15, 4.7,"Magnet 1",fontsize=16)
    ax.annotate("",[23.5, 4],[23.5, 5.5],arrowprops=dict(arrowstyle='<->'))
    ax.text(24, 4.7,"19.05 mm",fontsize=16)
    ax.annotate("",[20, 5.6],[23, 5.6],arrowprops=dict(arrowstyle='<->'))
    ax.text(20, 5.7 ,"15.875 mm",fontsize=16)


    ax.add_patch(patches.Rectangle((20, 2.7), 3, 0.3, facecolor='grey', alpha=0.2))
    ax.text(15, 2.8,"Magnet 2",fontsize=16)
    ax.annotate("",[23.5, 2.7],[23.5, 3.0],arrowprops=dict(arrowstyle='<->'))
    ax.text(24, 2.8,"3.0 mm",fontsize=16)
    ax.annotate("",[20, 2.6],[23, 2.6],arrowprops=dict(arrowstyle='<->'))
    ax.text(20, 2.3,"15.0 mm",fontsize=16)

    ax.annotate("",[23.5, 3],[23.5, 4],arrowprops=dict(arrowstyle='<->'))
    ax.text(24, 3.4, r"$d_\mathrm{sep}$",fontsize='x-large')

    ax.annotate("",[21.5, 5],[21.5, 4],arrowprops=dict(arrowstyle='->'))
    ax.text(21.8, 4.1,"F",fontsize=16)

    #plt.xscale('log')
    #plt.yscale('log')

    fig.set_tight_layout(True)
    #plt.tight_layout()
    plt.savefig('pics/big_magnet_spring.pdf')
    plt.show(block=False)
    raw_input("Hit enter")

    h_steps = 65
    x, F = calc_force_to_moving_magnet(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, Nr, Nphi, d_sep, h_steps)
    F_int = interp1d(x, F, kind='cubic')
    xe_int = interp1d(F, x, kind='cubic')
    xe = xe_int(-m*g)
    fig = plt.figure(facecolor='white', figsize=(16, 9))
    xp = np.linspace(-0.062,0.062, 126)
    # p3_ff = p3_k*xp + p3_k3*np.power(xp, 3)
    # p5_ff = p5_k*xp + p5_k3*np.power(xp, 3) + p5_k5*np.power(xp, 5)
    print "xe = %.3f mm" % (xe*1000)
    plt.axhline(0, -30, 30, color='black')
    # plt.plot(xp*1000, -p3_ff, '-', label="3rd order polyfit")
    # plt.plot(xp*1000, -p5_ff, '-', label="5th order polyfit")
    plt.plot(x*1000, F, '-', label="FEM")

    dd = 3e-3
    plt.plot(xe * 1000, F_int(xe), 'o')
    k = (F_int(xe + dd) - F_int(xe - dd)) / (2 * dd)
    ax = plt.gca()
    ax.text(xe * 1000, F_int(xe) + 0.5, r"$d_\mathrm{rest} = %.2f\,\mathrm{mm}$" % (xe * 1000), ha='center', fontsize=14)
    ax.text(xe * 1000, F_int(xe) - 0.8, r"$k = %.2f\,\mathrm{N/m}$" % (k), ha='center', fontsize=14)
    plt.plot([(xe - dd) * 1000, (xe + dd) * 1000], [F_int(xe - dd), F_int(xe + dd)])

    plt.xlabel('Moving magnet position in harvester [ mm ]')
    plt.ylabel('Restoring Force [ N ]')
    plt.legend(loc=2)
    plt.axis([-30, 30, -10.0, 10.0])
    fig.set_tight_layout(True)
    plt.savefig('pics/big_magnet_restoring_force.pdf')
    plt.show(block=False)
    raw_input("Hit enter")

if __name__ == "__main__":
    main()
