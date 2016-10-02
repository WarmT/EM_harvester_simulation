from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal
from scipy.signal import filtfilt

from magnet_force import *
import time

from magnet_flux import *

# import os, sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))


def falling_old(state, t, m, b):
    g = 9.819  # gravity in Helsinki

    x    = state[0]  # displacement
    xd   = state[1]  # velocity

    # compute acceleration xdd
    xdd = -g + b * xd / m

    # return the two state derivatives
    return [xd, xdd]


def falling(state, t, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, res, m, b):
    g = 9.819  # gravity in Helsinki

    x    = state[0]  # displacement
    xd   = state[1]  # velocity

    # calculate transduction factor (partial derivative of magnetic flux with respect to x)
    fluxdx = calc_flux_gradient(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, x)
    # compute acceleration xdd
    xdd = -g - b * xd / m - fluxdx * fluxdx / res * xd / m
    # time derivative of the magnetic flux is transduction factor times velocity
    fluxdt = fluxdx * xd

    # return all state derivatives
    return [xd, xdd, fluxdt]


def falling2(state, t, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, res, m, b, d):
    g = 9.819  # gravity in Helsinki

    x  = state[0]  # displacement
    xd = state[1]  # velocity

    # calculate transduction factor (partial derivative of magnetic flux with respect to x)
    fluxdx  = calc_flux_gradient(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, x)
    fluxdx -= calc_flux_gradient(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, x + d)
    # compute acceleration xdd
    xdd = -g - b * xd / m - fluxdx * fluxdx / res * xd / m
    # time derivative of the magnetic flux is transduction factor time velocity
    fluxdt = fluxdx * xd

#    print "t = %.2f, x = %.3f mm, fluxdx = %.2e V/(m/s), xd = %.2f m/s, xdd = %.5f m/s2, fluxdt = %.2e V/s" % (t*1000, x*1000, fluxdx, xd, xdd, fluxdt)

    # return all state derivatives
    return [xd, xdd, fluxdt]


def draw_coil_voltage(measfile, outfile, title, timediv, Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, d0, coil_R, load_R, two_coils, coil_d):

    start = time.time()

    data = np.genfromtxt(measfile, dtype='float', delimiter=';', skip_header=0)
    meas = data[:, 0]

#    if two_coils == False:
#        meas = data[:,0]
#    else:
#        meas = data

    # Filter requirements.
    order = 12
    fs = 5000.0       # sample rate, Hz
    cutoff = 100.0  # desired cutoff frequency of the filter, Hz
    stop_atten = 80.0
    ws = 2 * np.pi * cutoff / fs
    b, a = signal.cheby2(order, stop_atten, ws, 'low', analog=False)
    meas_filt = filtfilt(b, a, meas)
    tt_max = np.argmax(meas_filt, axis=None)
    tt_min = np.argmin(meas_filt, axis=None)

    tt = np.arange(0, timediv * 12, timediv * 12 / 1200)
    if tt.size > meas.size:
        tt = tt[0:-1]

#    d0 = 0.210 # initial position is 200 mm obove the coil
    v0 = 0.0   # initial velociti is 0 m/s
    flux0 = 0.0
    state0 = [d0, v0, flux0]
    t = np.arange(0.0, 0.24, 0.0001)

    m = 10.25 / 1000  # mass of the magnet is 10.25 g
    b = 0.00          # viscous damping is set to zero because both end of the tube are open in the fall test
    res = coil_R + load_R

    if two_coils:
        extra_args = (Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, res, m, b, coil_d)
        state = odeint(falling2, state0, t, extra_args)
    else:
        extra_args = (Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, res, m, b)
        state = odeint(falling, state0, t, extra_args)

    x = state[:, 0]     # displacement
    xd = state[:, 1]    # velocity
    flux = state[:, 2]  # flux

    mask = np.where((x < 0.04) & (x > -0.040))[0]
    # create masked time values
    t_mask = t[mask]

    # create time time differences between steps
    dt_mask = t_mask[1:] - t_mask[:-1]

    flux_mask = flux[mask]
    fluxd = flux_mask[1:] - flux_mask[:-1]
    volts = - fluxd / dt_mask * load_R / (coil_R + load_R)

    volts_max = np.argmax(volts, axis=None)
    volts_min = np.argmin(volts, axis=None)
#    c_t_mask = t_mask[1:] - (t_mask[volts_max] - tt[tt_max])
    c_t_mask = t_mask[1:]
    tt = tt - (tt[tt_max] - t_mask[volts_max])

    fig = plt.figure(facecolor='white', figsize=(16, 12))

    tmax = c_t_mask[volts_max] * 1000
    tmin = c_t_mask[volts_min] * 1000
    plt.axhline(0, 0, 1, color='black')
    plt.axvline(tmin, 0, 1, color='black')
    plt.axvline(tmax, 0, 1, color='black')

    plt.plot(c_t_mask * 1000, volts, 'm.', label="Calculated")
    plt.plot(tmax, volts[volts_max], 'mo')
    plt.plot(tmin, volts[volts_min], 'mo')

    plt.plot(tt * 1000, meas_filt, 'b', label="Measured")
    plt.plot(tt[tt_max] * 1000, meas_filt[tt_max], 'bo')
    plt.plot(tt[tt_min] * 1000, meas_filt[tt_min], 'bo')

    # set some fonts that allow upright \mu to be displayed
    plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})
    plt.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
    plt.rc('text', usetex=True)

    scale = plt.axis()
    plt.axis([tmax - 20, tmax + 10, scale[2], scale[3]])
    fig.set_tight_layout(True)
    plt.legend(loc=1)
    plt.ylabel("V")
    plt.xlabel("time (ms)")
    plt.title(title)
    plt.show(block=False)
    plt.savefig(outfile + ".pdf")
    plt.close()

    delta_t  =  t[1:] -  t[:-1]
#    delta_x  =  x[1:] -  x[:-1]
    delta_dx = xd[1:] - xd[:-1]
    xdd = delta_dx / delta_t
    t_c = (t[1:] + t[:-1]) / 2
#    fluxdx = (flux[1:] - flux[:-1]) / delta_x
    fluxdt = (flux[1:] - flux[:-1]) / delta_t
    volts = - fluxdt * load_R / (coil_R + load_R)

    fig = plt.figure(facecolor='white', figsize=(16, 20))
#    fig = plt.figure(facecolor='white', figsize=(8, 10))
    legend_location = 2

    t_c *= 1000
    t   *= 1000

    plt.subplot(6, 1, 1)
    plt.axhline(0, 0, 1, linestyle=':', color='black')
    plt.plot(t, x * 1000, label="Displacement")
    plt.ylabel("mm")
    plt.legend(loc=legend_location)
    plt.title(title)
    scale = plt.axis()
#    plt.axis([0, 250, scale[2], scale[3]])
    plt.axis([175, 230, scale[2], scale[3]])

    plt.subplot(6, 1, 2)
    plt.plot(t, xd, label="Velocity")
    plt.ylabel("m/s")
    plt.legend(loc=legend_location)
    scale = plt.axis()
#    plt.axis([0, 250, scale[2], scale[3]])
    plt.axis([175, 230, scale[2], scale[3]])

    plt.subplot(6, 1, 3)
    plt.axhline(0, 0, 1, linestyle=':', color='black')
    plt.plot(t_c, xdd, label="Acceleration")
    plt.ylabel(r"m/s^2")
    plt.legend(loc=legend_location)
    scale = plt.axis()
#    plt.axis([0, 250, scale[2], scale[3]])
    plt.axis([175, 230, scale[2], scale[3]])

    plt.subplot(6, 1, 4)
    plt.plot(t, flux, label=r"$\Phi_\mathrm{B}$")
    plt.ylabel(r"Wb-t")
    plt.legend(loc=legend_location)
    scale = plt.axis()
#    plt.axis([0, 250, scale[2], scale[3]])
    plt.axis([175, 230, scale[2], scale[3]])

    plt.subplot(6, 1, 5)
    plt.axhline(0, 0, 1, linestyle=':', color='black')
    plt.plot(t_c, fluxdt, label=r"$\frac{\mathrm d\Phi_\mathrm{B}}{\mathrm d t}$")
    plt.legend(loc=legend_location)
    scale = plt.axis()
#    plt.axis([0, 250, scale[2], scale[3]])
    plt.axis([175, 230, scale[2], scale[3]])

    tt *= 1000
    plt.subplot(6, 1, 6)
    plt.axhline(0, 0, 1, linestyle=':', color='black')
    plt.plot(t_c, volts, 'b', label=r"$V_\mathrm{load}$ (calculated)")
    plt.plot(tt, meas_filt, 'r', label=r"$V_\mathrm{load}$ (measured)")
    plt.ylabel("V")
    plt.xlabel("time [ ms ]")
    plt.legend(loc=legend_location)
    scale = plt.axis()
#    plt.axis([0, 250, scale[2], scale[3]])
    plt.axis([175, 230, scale[2], scale[3]])

    ax = [plt.subplot(6, 1, i + 1) for i in range(5)]
    for i, a in enumerate(ax):
        if i == 0:
            ax2 = a
        else:
            ax1 = ax2
            ax2 = a
#            nbins = len(ax1.get_xticklabels()) # added 
#            ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added #               
        a.set_xticklabels([])

    plt.subplots_adjust(wspace=0.01, hspace=0.15, left=0.05, right=0.98, top=0.97, bottom=0.05)

    plt.show(block=False)
    plt.savefig(outfile + "_all.pdf")
    end = time.time()
    raw_input("hit enter")
    print "Saved to %s. Elapsed time : %.2f seconds" % (outfile, end - start)


def construct_title(d_co, N, h_coil, r_i, r_o, coil_R, load_R, two_coils):
    title = r"$d_\mathrm{co} = %d\,$" + r"\textmu m, " + r"$N = %d,\,$" + \
            r"$h_\mathrm{coil} = %.2f\, \mathrm{mm},\,r_\mathrm{i} = %.2f$" + \
            r"$\,\mathrm{mm},\,r_\mathrm{o} = %.2f\,\mathrm{mm},\,$"
    if coil_R < 100:
        title = title + r"$R_\mathrm{coil}=%.2f\,\Omega,\,R_\mathrm{load}$"
    elif coil_R < 999:
        title = title + r"$R_\mathrm{coil}=%d\,\Omega,\,R_\mathrm{load}$"
    elif coil_R < 999e3:
        coil_R = coil_R / 1e3
        title = title + r"$R_\mathrm{coil}=%.2f\,\mathrm{k}\Omega,\,R_\mathrm{load}$"
    else:
        coil_R = coil_R / 1e6
        title = title + r"$R_\mathrm{coil}=%d\,\mathrm{M}\Omega,\,R_\mathrm{load}$"

    if load_R < 100:
        title = title + r"$=%.2f\,\Omega$"
    elif load_R < 999:
        title = title + r"$=%d\,\Omega$"
    elif load_R < 999e3:
        load_R = load_R / 1e3
        title = title + r"$=%.2f\,\mathrm{k}\Omega$"
    else:
        load_R = load_R / 1e6
        title = title + r"$=%d\,\mathrm{M}\Omega$"

    if two_coils:
        title = "Two coils, " + title
    else:
        title = "One coil, " + title

    title = title % (d_co * 1e6, N, h_coil * 1000, r_i * 1000, r_o * 1000, coil_R, load_R)
    return title


def main():

    a = 10
    f = 100

    r_i = 12.05 / 2 / 1000  # inner radius of the coil = 12.05/2 mm
    r_o = 25.8 / 2 / 1000   # outer radios of the coil = 25.3/2 mm
    h_coil = 6.0 / 1000         # coil height is 6.2 mm
    k_co = 0.55277
    d_co = 100e-6
#    m_D = 9.36 / 1000 # diameter of the magnet = 9.525 mm
    m_D = 9.525 / 1000  # diameter of the magnet = 9.525 mm
    r_mag = m_D / 2
    h_mag = 19.05 / 1000  # lenngth of the magnet = 19.05 mm
    m_Br = 1.31
    t0 = 0.45 * h_coil
    resistivity = 2.176
    N = 3000
    drop_h = 0.200


    measfile = "N3000_series_1M.csv"
    outfile = "N3000_series_1M"
    timediv = 5e-3
    N = 3000
    d_co = 100e-6
    two_coils = True
    coil_d = 20.0 / 1000
    coil_R = 783.0
    load_R = 1.0e6
    title = construct_title(d_co, N, h_coil, r_i, r_o, coil_R, load_R, two_coils)
    draw_coil_voltage(measfile, outfile, title, timediv, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, drop_h, coil_R, load_R, two_coils, coil_d)

    measfile = "N3000_100um.csv"
    outfile = "N3000_100um"
    timediv = 5e-3
    N = 3000
    d_co = 100e-6
#    h_coil = 6.2 / 1000
    two_coils = False
    title = construct_title(d_co, N, h_coil, r_i, r_o, coil_R, load_R, two_coils)
    draw_coil_voltage(measfile, outfile, title, timediv, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, drop_h, coil_R, load_R, two_coils, coil_d)

    measfile = "N1351_150um.csv"
    outfile = "N1351_150um"
    timediv = 5e-3
    N = 1351
    d_co = 150e-6
    r_o = 25.6 / 2 / 1000
    h_coil = 6.2 / 1000
    coil_R = 79.35
    load_R = 1.0e6
    two_coils = False
    title = construct_title(d_co, N, h_coil, r_i, r_o, coil_R, load_R, two_coils)
    draw_coil_voltage(measfile, outfile, title, timediv, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, drop_h, coil_R, load_R, two_coils, coil_d)

    measfile = "N800_200um.csv"
    outfile = "N800_200um"
    timediv = 5e-3
    N = 800
    d_co = 200e-6
    h_coil = 6.0 / 1000
    r_o = 26.0 / 2 / 1000
    coil_R = 26.65
    load_R = 1e6
    two_coils = False
    title = construct_title(d_co, N, h_coil, r_i, r_o, coil_R, load_R, two_coils)
    draw_coil_voltage(measfile, outfile, title, timediv, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, drop_h, coil_R, load_R, two_coils, coil_d)

    measfile = "N500_200um_75R.csv"
    outfile = "N500_200um_75R"
    timediv = 5e-3
    N = 500
    d_co = 200e-6
    h_coil = 12.1 / 1000
    r_o = 16.3 / 2 / 1000
    coil_R = 12.69
    load_R = 75.0
    two_coils = False
    title = construct_title(d_co, N, h_coil, r_i, r_o, coil_R, load_R, two_coils)
    draw_coil_voltage(measfile, outfile, title, timediv, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, drop_h, coil_R, load_R, two_coils, coil_d)

    measfile = "N500_200um_13R7.csv"
    outfile = "N500_200um_13R7"
    timediv = 5e-3
    N = 500
    d_co = 200e-6
    h_coil = 12.1 / 1000
    r_o = 16.3 / 2 / 1000
    coil_R = 12.69
    load_R = 13.7
    two_coils = False
    title = construct_title(d_co, N, h_coil, r_i, r_o, coil_R, load_R, two_coils)
    draw_coil_voltage(measfile, outfile, title, timediv, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, drop_h, coil_R, load_R, two_coils, coil_d)

    measfile = "N3000_series_783R.csv"
    outfile = "N3000_series_783R"
    r_i = 12.05 / 2 / 1000  # inner radius of the coil = 12.05/2 mm
    r_o = 25.8 / 2 / 1000   # outer radios of the coil = 25.3/2 mm
    h_coil = 6.0 / 1000         # coil height is 6.2 mm
    k_co = 0.55277
#    m_D = 9.36 / 1000 # diameter of the magnet = 9.525 mm
    m_D = 9.525 / 1000  # diameter of the magnet = 9.525 mm
    r_mag = m_D / 2
    h_mag = 19.05 / 1000  # lenngth of the magnet = 19.05 mm
    m_Br = 1.32
    t0 = 0.45 * h_coil

    timediv = 5e-3
    N = 3000
    d_co = 100e-6
    two_coils = True
    coil_d = 20.0 / 1000  # distance between coils (center-to-center)
    coil_R = 783.0
    load_R = 783.0
    title = r"Two coils in series"
    drop_h = 0.200

    title = construct_title(d_co, N, h_coil, r_i, r_o, coil_R, load_R, two_coils)
    draw_coil_voltage(measfile, outfile, title, timediv, m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, drop_h, coil_R, load_R, two_coils, coil_d)

#    raw_input("That's all folks!")


if __name__ == "__main__":
    main()
