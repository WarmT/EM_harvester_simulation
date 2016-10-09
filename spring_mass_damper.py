from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from magnet_force import *
from magnet_flux import *
import time

def spring_mass_damper(state, t, t0, d_rest, Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, res, m, b, s, F_spring, Fo, freq):
    g = 9.819  # gravity in Helsinki

    x    = state[0]  # x displacement
    xd   = state[1]  # x velocity
                     # 
    y    = state[3]  # y displacement
    yd   = state[4]  # y velocity
                     # 
    z    = state[5]  # z displacement
    zd   = state[6]  # z velocity
    
    w = 2 * np.pi * freq
#    y   =                          Fo * np.sin(2 * np.pi * freq * t)
#    yd  =   2 * np.pi * freq     * Fo * np.cos(2 * np.pi * freq * t)
#    ydd = -(2 * np.pi * freq)**2 * Fo * np.sin(2 * np.pi * freq * t)
    y   =  g / (w**2) * np.sin(w * t)
    yd  =  g /  w     * np.cos(w * t)
    ydd =  g          * np.sin(w * t)

    z = x - y
    zd = xd - yd

    # magnetic spring
    F_int = interp1d(s, F_spring, kind='cubic')
    # calculate transduction factor (partial derivative of magnetic flux with respect to x)
    fluxdx = calc_flux_gradient(Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, z-t0)
    # compute acceleration xdd
#    xdd = -g + F_int(z)/m + b * zd / m + fluxdx * fluxdx / res * zd / m
    xdd = -F_int(z-d_rest)/m - b * zd / m - fluxdx * fluxdx / res * zd / m
    # time derivative of the magnetic flux is transduction factor times velocity
    fluxdt = fluxdx * zd
    zdd = xdd - ydd
#    print "Y = %5.2f, x = %5.2f, y = %5.2f, z = %5.2f, xd = %5.2f, yd = %5.2f, zd = %5.2f, xdd = %5.2f, ydd = %5.2f, zdd = %5.2f " % (Fo * 1000, x * 1000, y * 1000, z * 1000, xd, yd, zd, xdd, ydd, zdd)

    # return all state derivatives
    #        0    1     2     3    4   5   6
    return [xd, xdd, fluxdt, yd, ydd, zd, zdd]

# def spring_mass_damper(state,t, m, b, s, F_spring, excit_force, freq):
#     x = state[0]
#     xd = state[1]

#     g = 9.819 # gravity in Helsinki
#     F_int = interp1d(s, F_spring, kind='cubic')
# #    print "x = %.3f" % (x*1000)
#     # compute acceleration xdd
#     try:
#         xdd = -F_int(x)/m - g - b*xd/m + excit_force/m*np.cos(2*np.pi*freq*t)
#     except:
#         print "x = %.5f" % (x)
    
#     # return the two state derivatives
#     return [xd, xdd]


def main():
    g = 9.819
    d0 = 0.029225
    # r_mag1 = 15.875e-3 / 2; h_mag1 = 19.05e-3; Br_mag1 = 1.31; m = 10.3e-3
    # r_mag2 = 15.000e-3 / 2; h_mag2 =  3.00e-3; Br_mag2 = 1.31

    r_mag1 = 4.7625e-3; h_mag1 = 19.05e-3; Br_mag1 = 1.31; m = 10.3e-3
    r_mag2 = 4.7625e-3; h_mag2 = 3.175e-3; Br_mag2 = 1.31
    d_co = 100e-6
    N = 3046
    h_coil = 7.05e-3
    r_i = 6.025e-3
    r_o = 12.1e-3

    Nr = 20
    Nphi = 60
    h_steps = 65
    s, F = calc_force_to_moving_magnet(r_mag1, h_mag1, Br_mag1, r_mag2, h_mag2, Br_mag2, Nr, Nphi, d0, h_steps)


    d_rest = -7.73e-3
#    state0 = [0.0, 0.0, 0.0, 0.0, 0.0, d_rest, 0.0]
    state0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #t = np.arange(0.0, 50.0, 0.1)
    dt = 0.001
    t = np.arange(0.0, 2, dt)

#    f_range = np.arange(8.033, 12, 0.01)
    freq = 7.0
    coil_R = 391.0
    load_R = 394
    res = coil_R + load_R
    b = 0.047
    print "b = %.4f" % b
    
    Wn = 2 * np.pi * freq
    Fo = g / Wn ** 2

    t0 = 8.65e-3
#    t0 = 0.0
#    for t0 in np.linspace(0.0, 20.0e-3, 21):
    extra_args = (t0, d_rest, Br_mag1, h_mag1, r_mag1, h_coil, r_i, r_o, N, d_co, res, m, b, s, F, Fo, freq)
    state = odeint(spring_mass_damper, state0, t, extra_args)

    x = state[:, 0]     # x displacement
    xd = state[:, 1]    # x velocity
    flux = state[:, 2]  # flux
    y = state[:, 3]     # y displacement
    yd = state[:, 4]    # y velocity
    z = state[:, 5]     # z displacement
    zd = state[:, 6]    # z velocity

    delta_t  =  t[1:] -  t[:-1]
    delta_dx = xd[1:] - xd[:-1]
    xdd = delta_dx / delta_t

    delta_dy = yd[1:] - yd[:-1]
    ydd = delta_dy / delta_t

    delta_dz = zd[1:] - zd[:-1]
    zdd = delta_dz / delta_t


    fluxd = flux[1:] - flux[:-1]
    t_c = (t[1:] + t[:-1]) / 2
    V = - fluxd / dt * load_R / (coil_R + load_R)

    V_rms = np.sqrt(np.mean(np.square(V)))
    print "V_rms = %.2f V" % (V_rms)


    """
    This part is for frequency sweep
    
    
    start = time.time()
    speed = []
    for freq in f_range:
        extra_args = (10.0, freq)
        state = odeint(MassSpring, state0, t, extra_args)
        speed = np.append(speed, (max(state[200:,1])-min(state[200:,1]))/2)
    
    end = time.time()
    print "odeint loop takes %.3f seconds to complete." % (end-start)

    plt.figure(facecolor='white', figsize=(16, 9))
    plt.plot(f_range, speed, label='speed (m/s)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (m/s)')
    plt.title('Mass-Spring System')
    plt.legend()
    #plt.legend(('$x$ (m)', '$\dot{x}$ (m/sec)'))
    plt.show(block=False)
    """

    fig = plt.figure(facecolor='white', figsize=(16, 9))

    plt.subplot(4,1,1)
    plt.hlines( Fo * 1000, 0, 1)
    plt.hlines(-Fo * 1000, 0, 1)
    plt.plot(t, x*1000, label = r"$x$")
    plt.plot(t, y*1000, label = r"$y$")
    plt.plot(t, z*1000, label = r"$z$")
    plt.legend(loc=1, fontsize='x-large')
    plt.ylabel("mm")
    plt.title(r"Base exitation $f_\mathrm{b}=%.2f\,\mathrm{Hz}$" % (freq))

    plt.subplot(4,1,2)
    plt.hlines( Fo * Wn, 0, 1)
    plt.hlines(-Fo * Wn, 0, 1)
    plt.plot(t, xd, label = r"$\dot{x}$")
    plt.plot(t, yd, label = r"$\dot{y}$")
    plt.plot(t, zd, label = r"$\dot{z}$")
    plt.legend(loc=1, fontsize='x-large')
    plt.ylabel("m/s")
#    plt.title('Velocity')

    plt.subplot(4,1,3)
    plt.hlines( Fo * Wn**2, 0, 1)
    plt.hlines(-Fo * Wn**2, 0, 1)
    plt.plot(t_c, xdd, label = r"$\ddot{x}$")
    plt.plot(t_c, ydd, label = r"$\ddot{y}$")
    plt.plot(t_c, zdd, label = r"$\ddot{z}$")
    plt.legend(loc=1, fontsize='x-large')
    plt.ylabel("m/s")
#    plt.title('Acceleration')

    plt.subplot(4,1,4)
    plt.plot(t_c, V, label = r"$V_\mathrm{load}$")
    plt.legend(loc=1, fontsize='x-large')
    plt.ylabel("V")
    plt.xlabel("time [ s ]")
#    plt.title('Coil voltage')
    
    ax = [plt.subplot(4, 1, i + 1) for i in range(3)]
    for i, a in enumerate(ax):
        if i == 0:
            ax2 = a
        else:
            ax1 = ax2
            ax2 = a
#            nbins = len(ax1.get_xticklabels()) # added 
#            ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added #               
        a.set_xticklabels([])


    fig.set_tight_layout(True)
#    plt.legend(('$x$ (m)', '$\dot{x}$ (m/sec)', 'voltage'))
    plt.show(block=False)

    raw_input("tadaa!!")
    plt.close()

if __name__ == "__main__":
    main()