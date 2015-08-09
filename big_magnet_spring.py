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

    d0 = 0.029225
    xs = 0.002225

    p3 = np.polyfit(s_meas, F_meas, 3)
    p5 = np.polyfit(s_meas, F_meas, 5)

    p5_alpha_1 = p5[4]
    p5_alpha_2 = p5[3]
    p5_alpha_3 = p5[2]
    p5_alpha_4 = p5[1]
    p5_alpha_5 = p5[0]


    p3_alpha_1 = p3[2]
    p3_alpha_2 = p3[1]
    p3_alpha_3 = p3[0]

    p3_k = 2*p3_alpha_1 + 4*d0*p3_alpha_2 + 6*d0*d0*p3_alpha_3
    p3_k3 = 2*p3_alpha_3

    p5_k = 2*p5_alpha_1 + 4*d0*p5_alpha_2 + 6*d0*d0*p5_alpha_3 + 8*d0*d0*d0*p5_alpha_4 + 10*d0*d0*d0*d0*p5_alpha_5
    p5_k3 = 2*p5_alpha_3 + 8*d0*p5_alpha_4 + 20*d0*d0*p5_alpha_5
    p5_k5 = 2*p5_alpha_5

    print "\nThird order polynomial fit:"
    print "p3_aplha_1 = %.3f, p3_alpha_2 = %.3f, p3_alpha_3 = %.3f" % (p3_alpha_1, p3_alpha_2, p3_alpha_3)
    print "d0 = %.4f mm, p3_k = %.3f N/m, p3_k3 = %.3f N/m^3" % (d0*1000, p3_k, p3_k3)
    print "\nFift order polynomial fit:"
    print "p5_aplha_1 = %.3f, p5_alpha_2 = %.3f, p5_alpha_3 = %.3f, p5_aplha_4 = %.3f, p5_alpha_5 = %.3f" % (p5_alpha_1, p5_alpha_2, p5_alpha_3, p5_alpha_4, p5_alpha_5)
    print "d0 = %.4f mm, p5_k = %.3f N/m, p5_k3 = %.3f N/m^3, p5_k5 = %.3f N/m^5" % (d0*1000, p5_k, p5_k3, p5_k5)

    pp3 = np.poly1d(p3)
    pp5 = np.poly1d(p5)

    h_range = np.arange(0.002, 0.062, 0.001)
    m1_r = 0.015875/2; m1_t = 0.01905; m1_Br = 1.2
    m2_r = 0.015000/2; m2_t = 0.00300; m2_Br = 1.2
    Nr = 20
    Nphi = 60
    start = time.time()
    F = calc_force_dist(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, h_range, Nr, Nphi)
    end = time.time()
    print "\nForce calculation takes %.4f seconds.\n" % (end-start)
    xp = np.linspace(0.002,0.062, 61)
    fig = plt.figure(facecolor='white', figsize=(16, 9))
    plt.axhline(0, 0, 60, color='black')
    plt.errorbar(np.multiply(s_meas,1000), F_meas, xerr=0.5, yerr=0.009819, label='Measurements')
    #plt.plot(np.multiply(s,1000), F, 'o', label='Measurements')
    plt.plot(np.multiply(xp, 1000), pp3(xp), '-', label="3rd order polyfit")
    plt.plot(np.multiply(xp, 1000), pp5(xp), '-', label="5th order polyfit")
    plt.plot(np.multiply(h_range,1000), F, '-', lw=2, label='FEM')
    plt.legend()
    plt.axis([0.0, 60, -2.0, 6.0])
    plt.xlabel('Separation Distance (mm)')
    plt.ylabel('Repulsion Force (N)')
    ax = plt.gca()
    ax.add_patch(patches.Rectangle((20, 4), 3, 1.5, facecolor='grey', alpha=0.2))
    ax.text(15, 4.7,"Magnet 1",fontsize=16)
    ax.annotate("",[23.5, 4],[23.5, 5.5],arrowprops=dict(arrowstyle='<->'))
    ax.text(24, 4.7,"19.05 mm",fontsize=16)


    ax.add_patch(patches.Rectangle((20, 2.7), 3, 0.3, facecolor='grey', alpha=0.2))
    ax.text(15, 2.8,"Magnet 2",fontsize=16)
    ax.annotate("",[23.5, 2.7],[23.5, 3.0],arrowprops=dict(arrowstyle='<->'))
    ax.text(24, 2.8,"3.0 mm",fontsize=16)
    ax.annotate("",[20, 2.6],[23, 2.6],arrowprops=dict(arrowstyle='<->'))
    ax.text(20, 2.3,"15.0 mm",fontsize=16)

    ax.annotate("",[23.5, 3],[23.5, 4],arrowprops=dict(arrowstyle='<->'))
    ax.text(24, 3.5,"Separation Distance",fontsize=16)

    ax.annotate("",[21.5, 5],[21.5, 4],arrowprops=dict(arrowstyle='->'))
    ax.text(21.8, 4.1,"F",fontsize=16)

    #plt.xscale('log')
    #plt.yscale('log')

    fig.set_tight_layout(True)
    #plt.tight_layout()
    plt.savefig('big_magnet_spring.png')
    plt.show()

    h_steps = 65
    x, F = calc_force_to_moving_magnet(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, Nr, Nphi, d0, h_steps)
    fig = plt.figure(facecolor='white', figsize=(16, 9))
    xp = np.linspace(-0.062,0.062, 126)
    p3_ff = p3_k*xp + p3_k3*np.power(xp, 3)
    p5_ff = p5_k*xp + p5_k3*np.power(xp, 3) + p5_k5*np.power(xp, 5)
    xe_int = interp1d(F, x, kind='cubic')
    xe = xe_int(m*g)
    print "xe = %.3f mm" % (xe*1000)
    plt.axhline(0, -30, 30, color='black')
    plt.plot(xp*1000, -p3_ff, '-', label="3rd order polyfit")
    plt.plot(xp*1000, -p5_ff, '-', label="5th order polyfit")
    plt.plot(x*1000, F, '-', label="FEM")
    plt.xlabel('Moving magnet position in harvester (mm)')
    plt.ylabel('Restoration Force (N)')
    plt.legend(loc=2)
    plt.axis([-30, 30, -10.0, 10.0])
    fig.set_tight_layout(True)
    plt.savefig('big_magnet_restoration_force.png')
    plt.show()

if __name__ == "__main__":
    main()
