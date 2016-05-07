from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from magnet_force import *
import time
from scipy.interpolate import interp1d


def main():
    m1_r = 0.009525/2
    m1_t = 0.01905
    m1_Br = 1.2
    m2_r = 0.009525/2
    m2_t = 0.003175
    m2_Br = 1.2
    Nr = 20
    Nphi = 20
    m = 0.01025
    g = 9.819 # gravitation in Helsinki
    
    h_range = np.linspace(0.001, 0.062, 63)
    F = calc_force(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, h_range, Nr, Nphi)
    
    fig = plt.figure(facecolor='white', figsize=(16, 9))
#    plt.axhline(0, 0, 60, color='black')
    plt.plot(np.multiply(h_range,1000), F, '-', lw=2, label='FEM')
#    plt.axis([0.0, 60, -2.0, 6.0])
    plt.xlabel('Separation Distance (mm)')
    plt.ylabel('Repulsion Force (N)')
    fig.set_tight_layout(True)
    plt.show()
    
    h_steps = 65
    fig = plt.figure(facecolor='white', figsize=(16, 9))
    plt.axhline(0, -20, 20, color='black')
    
    d0_range = np.linspace(0.01, 0.03, 11)
    for d0 in d0_range:
        start = time.time()
        x, F = calc_force_to_moving_magnet(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, Nr, Nphi, d0, h_steps)
        end = time.time()
        xe_int = interp1d(F, x, kind='cubic')
        xe = xe_int(m*g)
        print "d0 = %5.2f mm, xe = %6.3f mm" % (d0*1000, xe*1000)
#        print "Force calculation takes %.3f seconds." % (end-start)
        plt.plot(np.multiply(x,1000), F, '-', lw=2, label='d0 = %.2f mm' % (d0*1000))
    
    plt.legend(loc=2)
    plt.axis([-30, 30, -10, 10])
    plt.xlabel('Moving magnet position in harvester (mm)')
    plt.ylabel('Restoration Force (N)')
    fig.set_tight_layout(True)
    plt.savefig('small_magnet_spring_d0_sweep.png')
    plt.show()


if __name__ == "__main__":
    main()