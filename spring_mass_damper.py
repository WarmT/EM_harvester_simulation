from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from magnet_force import *
import time

def spring_mass_damper(state,t, m, b, s, F_spring, excit_force, freq):
    x = state[0]
    xd = state[1]

    g = 9.819 # gravity in Helsinki
    F_int = interp1d(s, F_spring, kind='cubic')
#    print "x = %.3f" % (x*1000)
    # compute acceleration xdd
    try:
        xdd = -F_int(x)/m - g - b*xd/m + excit_force/m*np.cos(2*np.pi*freq*t)
    except:
        print "x = %.5f" % (x)
    
    # return the two state derivatives
    return [xd, xdd]


def main():
    m = 28.49 / 1000
    g = 9.819
    d0 = 0.029225
    m1_r = 0.015875/2; m1_t = 0.01905; m1_Br = 1.2
    m2_r = 0.015000/2; m2_t = 0.00300; m2_Br = 1.2
    Nr = 20
    Nphi = 60
    h_steps = 65
    s, F = calc_force_to_moving_magnet(m1_r, m1_t, m1_Br, m2_r, m2_t, m2_Br, Nr, Nphi, d0, h_steps)


    state0 = [0.0, 0.0]
    #t = np.arange(0.0, 50.0, 0.1)
    t = np.arange(0.0, 1, 0.001)

#    f_range = np.arange(8.033, 12, 0.01)
    freq = 12.0
#    b = 2*0.035*m
    b = 2.0
    print "b = %.4f" % b
    excit_force = 6
    extra_args = (m, b, s, F, excit_force, freq)
    state = odeint(spring_mass_damper, state0, t, extra_args)

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
    x = state[:,0]
    xd = state[:,1]
    N = 1600
    fill = 0.6
    lam = 1.2*np.pi*0.015875*0.015875/4*0.01905*3/2
    y = 0.0022
    x1 = -0.00125
    x2 = 0.00475
    r1 = 0.01
    r2 = 0.015
    z11 = np.sqrt(r1*r1+(x1-x)*(x1-x))
    z12 = np.sqrt(r1*r1+(x2-x)*(x2-x))
    z21 = np.sqrt(r2*r2+(x1-x)*(x1-x))
    z22 = np.sqrt(r2*r2+(x2-x)*(x2-x))

    V = xd*N*fill*lam/((r2-r1)*(x2-x1))*(np.log(r1+z11)-r1/z11-np.log(r1+z12)+r1/z12-np.log(r2+z21)+r2/z21+np.log(r2+z22)-r2/z22)


    fig = plt.figure(facecolor='white', figsize=(16, 9))

    plt.subplot(3,1,1)
    plt.plot(t, state[:,0]*1000, label = "Magnet Displacement")
    plt.ylabel("mm")
    plt.title('Displacement')

    plt.subplot(3,1,2)
    plt.plot(t, state[:,1], label = "Magnet Velocity")
    plt.ylabel("m/s")
    plt.title('Velocity')

    plt.subplot(3,1,3)
    plt.plot(t, V, label = "Voltage")
    plt.ylabel("V")
    plt.xlabel("time (s)")
    plt.title('Coil voltage')

    fig.set_tight_layout(True)
#    plt.legend(('$x$ (m)', '$\dot{x}$ (m/sec)', 'voltage'))
    plt.show(block=False)

    raw_input("tadaa!!")

if __name__ == "__main__":
    main()