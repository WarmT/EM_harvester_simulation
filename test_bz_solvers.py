from __future__ import division
from scipy.special import ellipk, ellipe, ellipkinc, ellipeinc
from sympy import elliptic_k, elliptic_e, elliptic_pi
#from scipy import special
import numpy as np
from numba import jit
from scipy.interpolate import interp1d

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from magnet_flux import *

import sys
import timeit
import time


def Foelsch_example():
    coil_r =  0.06  #  6 cm
    hmag   =  0.40  # 40 cm
    r      =  0.09  #  9 cm
    x1     = -0.10  # 10 cm
    x2     =  0.30  # 30 cm

    print "###########################################"
    print "Foelsch example 1, case II, axial component"
    print "coil_r = %.2d m, hmag = %.2f m, r = %.2f m, x1 = %.2f m, x2 = %.2f m" % (coil_r, hmag, r, x1, x2)

    n = 4*coil_r*r/(coil_r+r)**2
    beta1 = (coil_r+r)**2 / ( (coil_r+r)**2 + x1*x1)
    beta2 = (coil_r+r)**2 / ( (coil_r+r)**2 + x2*x2)    
    m1 = n*beta1
    m2 = n*beta2
    K1 = ellipk(m1)
    E1 = ellipe(m1)
    K2 = ellipk(m2)
    E2 = ellipe(m2)
    
    sin2phi1=(1-n)/(1-m1)
    phi1=np.arcsin(np.sqrt(sin2phi1))
    sin2b1 = 1-m1
    b1 = np.arcsin(np.sqrt(sin2b1))
    Finc1 = ellipkinc(phi1, sin2b1)
    Einc1 = ellipeinc(phi1, sin2b1)
    A1 = np.pi/2 + K1*np.sqrt(1-beta1)*(1+np.sqrt(1-n))+Finc1*(K1-E1)-K1*Einc1
    B1 = 2*K1*np.sqrt(1-beta1) - A1
    
    sin2phi2=(1-n)/(1-m2) #    sin2phi2=0.04955 # This wrong value was given in Foelsch paper
    phi2=np.arcsin(np.sqrt(sin2phi2))
    sin2b2 = 1-m2
    b2 = np.arcsin(np.sqrt(sin2b2))
    Finc2 = ellipkinc(phi2, sin2b2)
    Einc2 = ellipeinc(phi2, sin2b2)
    A2 = np.pi/2 + K2*np.sqrt(1-beta2)*(1+np.sqrt(1-n))+Finc2*(K2-E2)-K2*Einc2
    B2 = 2*K2*np.sqrt(1-beta2) - A2
    
    print "n = %.3f, beta1 = %.5f, beta2 = %.5f, m1 = %.5f, m2 = %.5f" % (n, beta1, beta2, m1, m2)
    print "K1 = %.5f, E1 = %.5f, K2 = %.5f, E2 = %.5f" % (K1, E1, K2, E2)
    print "----"
    print "sin^2(phi_1)  = %.5f, phi_1 = %.3f rad = %.3f deg" % (sin2phi1, phi1, phi1/np.pi*180)
    print "sin^2(b_1)    = %.5f, b_1 = %.3f rad   = %.3f deg" % (sin2b1, b1, b1/np.pi*180)
    print "F(b_1, phi_1) = %.5f, E(b_1, phi_1)     = %.5f" % (Finc1, Einc1)
    print "A_1(n, beta1) = %.5f, B_1(n, beta1)     = %.5f" % (A1, B1)
    
    print "----"
    print "sin^2(phi_2)  = %.5f, phi_2 = %.3f rad = %.3f deg" % (sin2phi2, phi2, phi2/np.pi*180)
    print "sin^2(b_2)    = %.5f, b_2 = %.3f rad   = %.3f deg" % (sin2b2, b2, b2/np.pi*180)
    print "F(b_2, phi_2) = %.5f, E(b_2, phi_2)     = %.5f" % (Finc2, Einc2)
    print "A_2(n, beta2) = %.5f, B_2(n, beta2)     = %.5f" % (A2, B2)
    print "###########################################"
#    print "n = %.3f, beta1 = %.5f, beta2 = %.5f, m1 = %.5f, m2 = %.5f" % (n, beta1, beta2, m1, m2)
#    legendre(1.2, coil_r, hmag/2, r, x1)
#    legendre2(1.2, coil_r, hmag/2, r, x1)
#    print "###########################################"

def eval_err(table, val):
    err = 1e-6
    if table == val:
        return "OK"
    elif (val-table) < err:
        return "OK"
    else:
        return "NOK"

def test_bz_solvers():
    
    Br = 1.2
    coil_r = 6.0e-3
    a = 20.0e-3
    r = 9.0e-3

    zz = np.arange(-0.05, 0.05, 0.001)    
    tt_Foelsch1 = np.zeros(zz.size)
    tt_Foelsch2 = np.zeros(zz.size)
    tt_nasa = np.zeros(zz.size)
    tt_derby = np.zeros(zz.size)
    
    r = 9.1e-3
    hmag_per_2 = 20.0e-3
    hmag = 2.0*hmag_per_2
    z = hmag_per_2

    t_Foelsch1 = timeit.Timer(lambda: Foelsch1_axial(Br, coil_r, hmag_per_2, r, z))
    t_Foelsch2 = timeit.Timer(lambda: Foelsch2_axial(Br, coil_r, hmag_per_2, r, z))
    t_nasa = timeit.Timer(lambda: nasa_axial(Br, coil_r, hmag_per_2, r, z))
    t_derby = timeit.Timer(lambda: Derby_axial(Br, coil_r, hmag_per_2, r, z))


    no = 100000
    print "starting first sweep"
    start = time.time()
    for ind, z in enumerate(zz):
        tt_Foelsch1[ind] = t_Foelsch1.timeit(number=no)
        tt_Foelsch2[ind] = t_Foelsch2.timeit(number=no)
        tt_nasa[ind]     = t_nasa.timeit(number=no)
        tt_derby[ind]    = t_derby.timeit(number=no)
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)

    fig = plt.figure(facecolor='white', figsize=(17, 9))

    plt.subplot(1,2,1)
    plt.title("Number of iteratin per z: %d" % no, fontsize=12)
    plt.xlabel('z')
    plt.ylabel('Time (s)', fontsize=10)
    plt.plot(zz, tt_Foelsch1, label="Foelsch1")
    plt.plot(zz, tt_Foelsch2, label="Foelsch2")
    plt.plot(zz, tt_nasa, label="Nasa")
    plt.plot(zz, tt_derby, label="Derby")
    plt.legend(loc=1)
    
#    raw_input("Next radial sweep")
    print "next radial sweep"
    z = hmag_per_2 * 9 / 10
    print "z = %.5f" % (z)
    
    zz = np.arange(0.0, 0.02, 0.001)    
    tt_Foelsch1 = np.zeros(zz.size)
    tt_Foelsch2 = np.zeros(zz.size)
    tt_nasa = np.zeros(zz.size)
    tt_derby = np.zeros(zz.size)
    start = time.time()
    for ind, r in enumerate(zz):
        tt_Foelsch1[ind] = t_Foelsch1.timeit(number=no)
        tt_Foelsch2[ind] = t_Foelsch2.timeit(number=no)
        tt_nasa[ind]     = t_nasa.timeit(number=no)
        tt_derby[ind]    = t_derby.timeit(number=no)
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)    

    plt.subplot(1,2,2)
    plt.title("Number of iteratin per r: %d" % no, fontsize=12)
    plt.xlabel('r')
    plt.ylabel('Time (s)', fontsize=10)
    plt.plot(zz, tt_Foelsch1, label="Foelsch1")
    plt.plot(zz, tt_Foelsch2, label="Foelsch2")
    plt.plot(zz, tt_nasa, label="Nasa")
    plt.plot(zz, tt_derby, label="Derby")
#    plt.legend(loc=1)

    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/flux_density_speed.pdf")
    time.sleep(1)
    raw_input("Hit enter")
    plt.close()


def plot_bz():
    Br = 1.2
    coil_r = 6.0e-3
    a = 20.0e-3
    r = 9.0e-3

    zz = np.arange(-0.05, 0.05, 0.0001)    
    z_Foelsch1 = np.zeros(zz.size)
    z_Foelsch2 = np.zeros(zz.size)
    z_nasa = np.zeros(zz.size)
    z_derby = np.zeros(zz.size)
    z_analytic = np.zeros(zz.size)
    r = 0.0#01
    hmag_per_2 = 20.0e-3
    hmag = 2.0*hmag_per_2
    z = hmag_per_2
    start = time.time()
    for ind, z in enumerate(zz):
        z_Foelsch1[ind] = Foelsch1_axial(Br, coil_r, hmag_per_2, r, z)
        z_Foelsch2[ind] = Foelsch2_axial(Br, coil_r, hmag_per_2, r, z)
        z_nasa[ind]     = nasa_axial(Br, coil_r, hmag_per_2, r, z)
        z_derby[ind]    = Derby_axial(Br, coil_r, hmag_per_2, r, z)
        z_analytic[ind] = Br/2*((z+hmag_per_2)/np.sqrt((z+hmag_per_2)*(z+hmag_per_2)+coil_r*coil_r)-(z-hmag_per_2)/np.sqrt((z-hmag_per_2)*(z-hmag_per_2)+coil_r*coil_r))

    fig = plt.figure(facecolor='white', figsize=(17, 9))
    plt.subplot(3, 1, 1)
    plt.plot(zz*1000, z_Foelsch1, label="Foelsch1")
    plt.plot(zz*1000, z_Foelsch2, label="Foelsch2")
    plt.plot(zz*1000, z_nasa, label = "Nasa")
    plt.plot(zz*1000, z_derby, label = "Derby")
    plt.plot(zz*1000, z_analytic, label = "Analytic")
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.axvline(1.2, 0, 1, color='black')
 #   plt.xlabel("z [mm]")
    plt.ylabel(r"$B_z\,\,[\,T\,]$")
    plt.title(r'Magnetic flux density $B_z$ ($r=0$)')
    plt.legend(loc=1)
    scale = plt.axis()
    plt.axis([-50, +50, scale[2], scale[3]])

    err_Foelsch1 = np.divide((z_Foelsch1 - z_analytic), z_analytic)
    err_Foelsch2 = np.divide((z_Foelsch2 - z_analytic), z_analytic)
    err_nasa     = np.divide((z_nasa - z_analytic), z_analytic)
    err_derby    = np.divide((z_derby - z_analytic), z_analytic)

    plt.subplot(3, 1, 2)
    plt.plot(zz*1000, err_Foelsch1, label="(Foelsch1 - Analytic) / Analytic")
    plt.plot(zz*1000, err_Foelsch2, label="(Foelsch2 - Analytic) / Analytic")
#    plt.plot(zz*1000, z_analytic, label = "Analytic")
    plt.plot(zz*1000, err_nasa, label = "(Nasa - Analytic) / Analytic")
    plt.plot(zz*1000, err_derby, label = "(Derby - Analytic) / Analytic")
#    plt.xlabel("z [mm]")
#    plt.ylabel(r"$B_z\,\,[\,T\,]$")
#    plt.title(r'Magnetic flux density $B_z$ ($r=0$)')
    plt.legend(loc=1)
    scale = plt.axis()
    plt.axis([-50, +50, scale[2], scale[3]])

    plt.subplot(3, 1, 3)
    plt.plot(zz*1000, err_Foelsch1, label="(Foelsch1 - Analytic) / Analytic")
    plt.plot(zz*1000, err_Foelsch2, label="(Foelsch2 - Analytic) / Analytic")
#    plt.plot(zz*1000, z_analytic, label = "Analytic")
    plt.plot(zz*1000, err_nasa, label = "(Nasa - Analytic) / Analytic")
    plt.plot(zz*1000, err_derby, label = "(Derby - Analytic) / Analytic")
    plt.xlabel("z [mm]")
#    plt.ylabel(r"$B_z\,\,[\,T\,]$")
#    plt.title(r'Magnetic flux density $B_z$ ($r=0$)')
#    plt.legend(loc=1)
    scale = plt.axis()
    plt.axis([-50, +50, -3e-14, 3e-14])

    ax = [plt.subplot(3, 1, i + 1) for i in range(2)]
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

    plt.show(block=False)
    plt.savefig("Axial_flux1.pdf")

    Foelsch1_err = max(z_Foelsch1 - z_analytic)
    Foelsch2_err = max(z_Foelsch2 - z_analytic)
    derby_err = max(z_derby - z_analytic)
    nasa_err = max(z_nasa - z_analytic)
    
    print "Foelsch1_err = %e Foelsch2_err = %e, Derby_err = %e, Nasa_err = %e" % (Foelsch1_err, Foelsch2_err, derby_err, nasa_err)
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)
    
    raw_input("Next radial sweep")
    z = hmag_per_2 * 9 / 10
    print "z = %.5f" % (z)
    plt.close()
    
    fig = plt.figure(facecolor='white', figsize=(17, 9))
    zz = np.arange(0.0, 0.02, 0.00002)
    z_Foelsch1 = np.zeros(zz.size)
    z_Foelsch2 = np.zeros(zz.size)
    z_nasa = np.zeros(zz.size)
    z_derby = np.zeros(zz.size)
    start = time.time()
    for ind, r in enumerate(zz):
        z_Foelsch1[ind] = Foelsch1_axial(Br, coil_r, hmag_per_2, r, z)
        z_Foelsch2[ind] = Foelsch2_axial(Br, coil_r, hmag_per_2, r, z)
        z_nasa[ind]     = nasa_axial(Br, coil_r, hmag_per_2, r, z)
        z_derby[ind]    = Derby_axial(Br, coil_r, hmag_per_2, r, z)

    plt.subplot(3, 1, 1)
    plt.plot(zz*1000, z_Foelsch1, label="Foelsch1")
    plt.plot(zz*1000, z_Foelsch2, label="Foelsch2")
    plt.plot(zz*1000, z_nasa, label = "Nasa")
    plt.plot(zz*1000, z_derby, label = "Derby")
    plt.axhline(0, 0, 1.0, color='black')
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlabel("r [mm]")
    plt.ylabel(r"$B_z\,\,[\,T\,]$")
    plt.title(r'Magnetic flux density $B_z$ ($z=%.1f\,mm$)' % (z*1000))
    plt.legend(loc=1)

    err_Foelsch1 = np.divide((z_Foelsch1 - z_derby), z_derby)
    err_Foelsch2 = np.divide((z_Foelsch2 - z_derby), z_derby)
    err_nasa     = np.divide((z_nasa - z_derby), z_derby)

    plt.subplot(3, 1, 2)
    plt.plot(zz*1000, err_Foelsch1, label="(Foelsch1 - Derby) / Derby")
    plt.plot(zz*1000, err_Foelsch2, label="(Foelsch2 - Derby) / Derby")
    plt.plot(zz*1000, err_nasa, label = "(Nasa - Derby) / Derby")
#    plt.axhline(0, 0, 1.0, color='black')
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlabel("r [mm]")
#    plt.ylabel(r"$B_z\,\,[\,T\,]$")
#    plt.title(r'Magnetic flux density $B_z$ ($z=%.1f\,mm$)' % (z*1000))
    plt.legend(loc=1)

    plt.subplot(3, 1, 3)
    plt.plot(zz*1000, err_Foelsch1, label="(Foelsch1 - Derby) / Derby")
    plt.plot(zz*1000, err_Foelsch2, label="(Foelsch2 - Derby) / Derby")
    plt.plot(zz*1000, err_nasa, label = "Nnasa - Derby) / Derby")
#    plt.axhline(0, 0, 1.0, color='black')
#    plt.xscale('log')
#    plt.yscale('log')
    plt.xlabel("r [mm]")
#    plt.ylabel(r"$B_z\,\,[\,T\,]$")
#    plt.title(r'Magnetic flux density $B_z$ ($z=%.1f\,mm$)' % (z*1000))
#    plt.legend(loc=1)
    scale = plt.axis()
    plt.axis([scale[0], scale[1], -1e-13, 1e-13])

    ax = [plt.subplot(3, 1, i + 1) for i in range(2)]
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
    plt.show(block=False)
    plt.savefig("Axial_flux2.pdf")
    
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)
    raw_input("end of plotting")
    plt.close()
    
    m_h = 2*hmag_per_2
    m_r = r
    step = 0.001
#    dd = np.arange(-0.03, 0.03+step, step)    
    parts = 19
    steps  = 200
    steps2 = int(steps / 2)
    ymax = m_h*1.2
    xmax = ymax

    Y, X = np.mgrid[-ymax:ymax:200j, -xmax:xmax:200j]
    B = np.zeros((steps,steps))
    
    Br = 1.2

    for i in range(steps):
        for j in range(steps2):
            Bz_axial  = nasa_axial(Br, m_r, m_h/2, X[i][steps2+j], Y[i][steps2+j])
            B[i][steps2+j] = -Bz_axial
            B[i][steps2-j] = -Bz_axial

    fig = plt.figure(facecolor='white', figsize=(10, 10))
    ax = plt.gca()

    CS = plt.contour(X*1000, Y*1000, B, 30, colors='k')
    plt.clabel(CS, fontsize=9, inline=1)

    ax.annotate("",[0, -10],[0, 10],arrowprops=dict(lw=5, color='blue', arrowstyle='<-'))

    xmax *= 950
    ymax *= 950
    plt.axis([-xmax, xmax, -ymax, ymax])
    plt.xlabel("(mm)")
    plt.ylabel("(mm)")
    fig.set_tight_layout(True)
    
    plt.show(block=False)    
    plt.savefig("Axial_flux3.pdf")
    raw_input("tadaa!")
    plt.close()

#    fig = plt.figure(facecolor='white', figsize=(17, 9))
#    zz = np.arange(0.0, 0.02, 0.0001)
#    z_Foelsch1 = np.zeros(zz.size)
#    z_Foelsch2 = np.zeros(zz.size)
#    z_nasa = np.zeros(zz.size)
#    z_derby = np.zeros(zz.size)
#    for ind, r in enumerate(zz):
#        z_Foelsch1[ind] = Foelsch1_axial(Br, coil_r, hmag_per_2, r, z)
#        z_Foelsch2[ind] = Foelsch2_axial(Br, coil_r, hmag_per_2, r, z)
#        z_nasa[ind]     = nasa_axial(Br, coil_r, hmag_per_2, r, z)
#        z_derby[ind]    = Derby_axial(Br, coil_r, hmag_per_2, r, z2)
#
#    plt.plot(zz, z_Foelsch1, label="Foelsch1")
#    plt.plot(zz, z_Foelsch2, label="Foelsch2")
#    plt.plot(zz, z_derby, 'o', label = "Derby")
#    plt.plot(zz, z_nasa, label = "Nasa")
#    plt.axhline(0, 0, 1.0, color='black')
##    plt.xscale('log')
##    plt.yscale('log')
#    plt.legend(loc=1)
#    plt.show(block=False)    
#    
#    raw_input("end of plotting")
#    plt.close()


def flux_linkage_nasa_slice(Br, mag_h, mag_r, coil_h, coil_r1, coil_r2, k_co, d_co, z, parts):#, parts2):
    Nz = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
    Nr  = 2 * (coil_r2 - coil_r1) / (d_co * np.sqrt(np.pi/k_co))
    dN = Nz * Nr / (parts * parts)
    phi = 0.0
    dr = (coil_r2 - coil_r1) / parts
    dr2 = dr / 2
    dz = coil_h / parts
 
    r = 0
    dphi = 0.0

    Bz1 = np.zeros(parts)
    Bz2 = np.zeros(parts)
    Bz3 = np.zeros(parts)
    phi1 = np.zeros(parts)
    phi2 = np.zeros(parts)
    phi3 = np.zeros(parts)
    r1 = np.zeros(parts)
    r2 = np.zeros(parts)
    r3 = np.zeros(parts)
    
    dr = mag_r / parts
    dr2 = dr / 2
    for i in xrange(parts):
        Bz = nasa_axial(Br, mag_r, mag_h/2, r+dr2, z)
        Bz1[i] = Bz
        r1[i] = r
        ddphi = Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
        dphi += ddphi
        phi += dphi
        phi1[i] = phi
        r += dr
    print "r1 = "
    print r1*1000
    print "Bz1 = "
    print Bz1
    
#    print "0       -> mag_r   : r = %6.3f (mag_r   = %6.3f)" % (r*1000, mag_r*1000)
        
    dr = (coil_r1 - mag_r) / parts
    dr2 = dr / 2
    for i in xrange(parts):
        Bz = nasa_axial(Br, mag_r, mag_h/2, r+dr2, z)
        Bz2[i] = Bz
        r2[i] = r
        ddphi = Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
        dphi += ddphi
        phi += dphi
        phi2[i] = phi
        print "Bz = %.5f, ddphi = %.5f, dphi = %.5f, phi = %.5f" % (Bz, ddphi, dphi, phi)
        r += dr
#    print "mag_r   -> coil_r1 : r = %6.3f (coil_r1 = %6.3f)" % (r*1000, coil_r1*1000)

    dr = (coil_r2 - coil_r1) / parts
    dr2 = dr / 2
    for i in xrange(parts):
        Bz = nasa_axial(Br, mag_r, mag_h/2, r+dr2, z)
        Bz3[i] = Bz
        r3[i] = r
        ddphi = dN * Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
        dphi += ddphi
        phi += dphi
        phi3[i] = phi
        r += dr
#    print "coil_r1 -> coil_r2 : r = %6.3f (coil_r2 = %6.3f)" % (r*1000, coil_r2*1000)


    fig = plt.figure(facecolor='white', figsize=(17, 9))
    
    plt.subplot(2,1,1)
    plt.axvline(mag_r*1000, 0, 1, color='black')
    plt.axvline(coil_r1*1000, 0, 1, color='black')
    plt.axvline(coil_r2*1000, 0, 1, color='black')
    plt.axhline(0, 0, 1, color='black')

    plt.xlabel("z [mm]")
    plt.ylabel(r"$B_z\,\, [\,Wb\,]$")
    plt.title('Flux (z = %.2f mm)' % (z*1000))
    plt.step(r1*1000, Bz1, where='post', label="firts")
    plt.plot(r1*1000, Bz1, 'o', label="firts")
    plt.step(r2*1000, Bz2, where='post', label="second")
    plt.plot(r2*1000, Bz2, 'o', label="second")
    plt.step(r3*1000, Bz3, where='post', label="third")
    plt.plot(r3*1000, Bz3, 'o', label="third")
#    plt.plot(dd*1000, phi, label = "Elliptic")
    scale = plt.axis()
    plt.axis([0, coil_r2*1000+1, scale[2], scale[3]])
    plt.legend(loc=1)
    plt.show(block=False)    
#    raw_input("end of plotting")
#    plt.close()

#    fig = plt.figure(facecolor='white', figsize=(17, 9))
    plt.subplot(2,1,2)
    plt.axvline(mag_r*1000, 0, 1, color='black')
    plt.axvline(coil_r1*1000, 0, 1, color='black')
    plt.axvline(coil_r2*1000, 0, 1, color='black')
    plt.axhline(0, 0, 1, color='black')

    plt.xlabel("z [mm]")
    plt.ylabel(r"$\lambda\,\, [\,Wb-t\,]$")
    plt.title('Flux linkage (z = %.2f mm)' % (z*1000))
    plt.step(r1*1000, phi1, where='post', label="firts")
    plt.plot(r1*1000, phi1, 'o', label="firts")
    plt.step(r2*1000, phi2, where='post', label="second")
    plt.plot(r2*1000, phi2, 'o', label="second")
    plt.step(r3*1000, phi3, where='post', label="third")
    plt.plot(r3*1000, phi3, 'o', label="third")
#    plt.plot(dd*1000, phi, label = "Elliptic")
    scale = plt.axis()
    plt.axis([0, coil_r2*1000+1, scale[2], scale[3]])
    plt.legend(loc=1)
    plt.show(block=False)    
    raw_input("end of plotting")
    plt.close()




def test_flux_linkage():
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm 
    coil_r2 = 25.8 / 2 / 1000  # outer radios of the coil = 25.3/2 mm
    coil_h = 6.2 / 1000 # coil height is 6.2 mm

    # Parameters for prototype 3
    N = 3000  # number of turns
#    m_D = 9.35 / 1000 # diameter of the magnet = 9.525 mm
    m_D = 9.525 / 1000 # diameter of the magnet = 9.525 mm
    m_r = m_D/2
    m_h = 19.05 / 1000 # lenngth of the magnet = 19.05 mm
    m_Br = 1.2
    
    dd = np.arange(-0.05, 0.05, 0.001)
    phi_Foelsch2 = np.zeros(dd.size)
    phi_nasa = np.zeros(dd.size)
    phi_Derby = np.zeros(dd.size)
    
    k_co = 0.55277
    d_co = 100e-6

    for parts in np.arange(1, 51):
        Nz = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
        Nr  = 2 * (coil_r2 - coil_r1) / (d_co * np.sqrt(np.pi/k_co))
        Ndz = Nz / parts
        Ndr = Nr / parts
#        dz = coil_h / parts
#        dr = (coil_r2 - coil_r1) / parts
        dN = Nz * Nr / (parts * parts)
#        print "i = %2d, parts = %2d, Nz = %.2f, Nr = %.2f, N = %4d, Ndz = %5.2f, Ndr = %5.2f, dN = %.2f" % (i, parts, Nz, Nr, np.round(Nr*Nz), Ndz, Ndr, dN)

    parts = 21
    
    plt.ion()
    fig = plt.figure(facecolor='white', figsize=(17, 9))
    start = time.time()
    for parts in np.arange(1, 32):
        for ind, d in enumerate(dd):
#           phi_Foelsch2[ind] = flux_linkage_Foelsch2_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, 11)#, 50)
           phi_nasa[ind] = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 50)
#                            flux_linkage_Derby_axial(Br, mag_h, mag_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts): #, parts2):
#            phi_Derby[ind] = flux_linkage_Derby_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 50)
        plt.plot(dd, phi_nasa, label = "Flux linkage (parts = %2d)" % parts)
#        plt.plot(dd, phi_Derby, label = "Flux linkage (parts = %2d)" % parts)
        plt.legend(loc=1)
#        plt.show()
        plt.draw()
        raw_input("parts = %2d" % (parts))
        if parts%6 == 0:
            plt.clf()
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)    

    
#    plt.plot(dd, phi_Foelsch2, label = "Foelch2 Flux linkage")
#    plt.plot(dd, phi_nasa, label = "Nasa Flux linkage")
#    plt.plot(dd, phi_Derby, 'o', label = "Derby Flux linkage")
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.legend(loc=1)
#    plt.show(block=False)    
    
    raw_input("end of plotting")
#    plt.close()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def test_flux_linkage2():
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm 
    coil_r2 = 25.8 / 2 / 1000  # outer radios of the coil = 25.3/2 mm
    coil_h = 6.2 / 1000 # coil height is 6.2 mm

    # Parameters for prototype 3
    N = 3000  # number of turns
#    m_D = 12 / 1000 # diameter of the magnet = 9.525 mm
    m_D = 9.525 / 1000 # diameter of the magnet = 9.525 mm
#    m_D = 9.36 / 1000 # diameter of the magnet = 9.525 mm
    m_r = m_D/2
    m_h = 19.05 / 1000 # lenngth of the magnet = 19.05 mm
    m_Br = 1.32
    
    k_co = 0.55277
    d_co = 100e-6

    step = 0.0001
    step = coil_h / 100

    Nz_float = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
    Nz = int(round(Nz_float))
    step = coil_h / (Nz)
    
#    dd = np.arange(-0.03, 0.03+step, step)
#    dd = np.arange(-2*m_h, 2*m_h+step, step)
    offset = step*1/100
    offset = 0.0
    dd = np.arange(-3*m_h+offset, 3*m_h+step+offset, step)
    phi_nasa_orig = np.zeros(dd.size)
    phi_nasa = np.zeros(dd.size)
    phi_nasa_all = np.zeros(dd.size)
    phi_Derby = np.zeros(dd.size)
    phi_Foelsch2 = np.zeros(dd.size)
    
    filename = "falling_magnet/elliptic_fl.csv"
    result_csv_file = open(filename, "w")
    result_csv_file.write("z;fl;\n")
    
    parts = 25
    step_no = dd.size
    print "Number of steps: %d" % (step_no)
    start = time.time()
    progress = 0
    percentile = int(round(dd.size/100))
    for ind, d in enumerate(dd):
        if ind % percentile == 0:
            progress += 1
            sys.stdout.write("Calculation progress: %d%%   \r" % (progress) )
            sys.stdout.flush()
#            print "%3d%% done." % (percent)
#        tmp = flux_linkage_nasa_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
#        phi_nasa_orig[ind] = tmp
#        tmp = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
#        phi_nasa[ind] = tmp

        tmp = flux_linkage_Derby_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
        phi_nasa_all[ind] = tmp
        result_csv_file.write("%.8f;%.8f;\n" % (d, tmp))

#        tmp = flux_linkage_Derby_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
#        phi_Derby[ind] = tmp        
#        tmp = flux_linkage_Foelsch2_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
#        phi_Foelsch2[ind] = tmp
    end = time.time()
    print "Elapsed time : %.2f seconds (%.2f ms/step)" % (end-start, (end-start)/step_no*1000)
    
    dz_flux_all = ( phi_nasa_all[1:] - phi_nasa_all[:-1] ) / step
    dz = dd[:-1] + step/2    

    result_csv_file.close()

#    print "z = % .4f, Bz = %.5f," % (dd[0], flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, dd[0], parts))
#    print "z = % .4f, Bz = %.5f," % (dd[-1], flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, dd[-1], parts))

    data = np.genfromtxt("flux_N_up.csv", dtype='float', delimiter=';', skip_header = 0)
    z = data[:,0]/1000
    z_step = z[1]-z[0]
    z = z+m_h/2
    fl = data[:,1]
    fl_dt = ( fl[1:] - fl[:-1] ) / z_step
    f_int = interp1d(z, fl, kind='cubic')
#    radius = phi_nasa/fl
#    print "radius = "
#    print radius
    
    print "z = % .4f, Bz = %.5f," % (z[0], fl[0])
    print "z = % .4f, Bz = %.5f," % (z[-1], fl[-1])
    print
    print "z = % .4f, Bz = %.5f," % (z[1], fl[1])
    print "z = % .4f, Bz = %.5f," % (z[-2], fl[-2])

#    fig = plt.figure(facecolor='white', figsize=(17, 9))
#    plt.xlabel("z [mm]")
#    plt.ylabel(r"$\lambda\,\, [\,Wb-t\,]$")
    fig, ax1 = plt.subplots(facecolor='white', figsize=(17, 9))
    ax1.set_xlabel("z [mm]")
    ax1.set_ylabel(r"$\lambda\,\, [\,Wb-t\,]$", color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    plt.title('Flux linkage')
    ax1.plot(z*1000, fl, label="FEMM")
#    ax1.plot(dd*1000, phi_nasa_orig, label = "nasa_orig")
#    ax1.plot(dd*1000, phi_nasa, label = "nasa")
    ax1.plot(dd*1000, phi_nasa_all, label = "nasa_all")
#    plt.plot(dd*1000, phi_Derby, 'o', label = "Derby")
#    ax1.plot(dd*1000, phi_Foelsch2, 'x', label = "Foelsch2")
    
    plt.axvline(-m_h*1000/2, 0, 1, color='black')
    plt.axvline( m_h*1000/2, 0, 1, color='black')
    plt.axvline(0, 0, 1, color='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel('V/(m/s)', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
#    ax2.plot(dz[1:-1]*1000, dz_flux2, label = "Flux gradient (smooth)")
#    ax2.plot(dz*1000, dz_flux2, label = "Flux gradient (smooth)")
#    ax2.plot(dz*1000, dz_flux1, 'r', label = "Flux gradient")
    ax2.plot((z[1:]-z_step/2)*1000, fl_dt, 'r', label = "Flux gradient (FEMM)")
#    ax2.plot(dz*1000, dz_flux_orig, 'b-.', label = "Flux gradient (orig)")
    ax2.plot(dz*1000, dz_flux_all, 'g', label = "Flux gradient (nasa_all)")
    ax2.plot(dz*1000, dz_flux_all, 'g.')
#    plt.axvline(m_h/2*1000, 0, 1, color='black')
#    plt.axvline(-m_h/2*1000, 0, 1, color='black')

#    step = 0.0003
#    d = -m_h/2-step*2; y1 = flux_linkage_nasa_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
#    d = -m_h/2-step*1; y2 = flux_linkage_nasa_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
##    d = -m_h/2-step*0; y3 = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
#    d = -m_h/2+step*1; y4 = flux_linkage_nasa_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
#    d = -m_h/2+step*2; y5 = flux_linkage_nasa_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
#    tmp = (y5-y1)*5/4/step + (y5+y4-y2-y1)*5/6/step
#    tmp = tmp / 10.0
#    ax2.plot(-m_h/2*1000, tmp, 'k+', markersize=30, fillstyle='full')



    scale = plt.axis()
    plt.axis([-30, 30, scale[2], scale[3]])
#    plt.axis([-15, 0, scale[2], scale[3]])
    ax1.legend(loc=3)
    ax2.legend(loc=4)
#    plt.show()    
    plt.show(block=False)    
    raw_input("end of plotting")

    plt.savefig("flux_linkage.pdf")
    plt.close()


def test_flux_linkage3():
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm 
    coil_r2 = 25.8 / 2 / 1000  # outer radios of the coil = 25.3/2 mm
    coil_h = 6.2 / 1000 # coil height is 6.2 mm

    # Parameters for prototype 3
    N = 3000  # number of turns
    m_D = 9.35 / 1000 # diameter of the magnet = 9.525 mm
    m_r = m_D/2
    m_h = 19.05 / 1000 # lenngth of the magnet = 19.05 mm
    m_Br = 1.2
    

    k_co = 0.55277
    d_co = 100e-6
    parts = 7


    start = time.time()
    dd = np.arange(-0.01, 0.01, 0.001)
    for d in dd:
        phi = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
        print "d = % .4f, phi = % .5f, diff = %.4e" % (d, phi, d+0.001)
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)    

    d = -0.001 + 4.0766e-17
    phi = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
    print "d = % .4f, phi = % .5f" % (d, phi)
    
    d = -0.001 - 4.0766e-17
    phi = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
    print "d = % .4f, phi = % .5f" % (d, phi)
    
    d = -0.001
    phi = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
    print "d = % .4f, phi = % .5f" % (d, phi)
    
    d = 0.0
    phi = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
    print "d = % .4f, phi = % .5f" % (d, phi)
    
    d = 0.001
    phi = flux_linkage_nasa_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)#, 51)
    print "d = % .4f, phi = % .5f" % (d, phi)



def draw_flux_lines():
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm 
    coil_r2 = 25.8 / 2 / 1000  # outer radios of the coil = 25.3/2 mm
    coil_h = 6.2 / 1000 # coil height is 6.2 mm

    # Parameters for prototype 3
    N = 3000  # number of turns
#    m_D = 12 / 1000 # diameter of the magnet = 9.525 mm
#    m_D = 9.525 / 1000 # diameter of the magnet = 9.525 mm
    m_D = 9.36 / 1000 # diameter of the magnet = 9.525 mm
    m_r = m_D/2
    m_h = 19.05 / 1000 # lenngth of the magnet = 19.05 mm
    m_Br = 1.2
    
    step = 0.001
    dd = np.arange(-0.03, 0.03+step, step)
    phi = np.zeros(dd.size)
    
    k_co = 0.55277
    d_co = 100e-6
    parts = 19

    steps  = 200
    steps2 = int(steps / 2)
    ymax = 50/1000
    xmax = 50/1000
    Y, X = np.mgrid[-ymax:ymax:200j, -xmax:xmax:200j]
    XX = np.zeros((steps,steps))
    YY = np.zeros((steps,steps))
    
    Br = 1.2

    for i in range(steps):
        for j in range(steps2):
            Bz_axial  = nasa_axial(Br, m_r, m_h/2, X[i][steps2+j], Y[i][steps2+j])
            Bz_radial = nasa_radial(Br, m_r, m_h/2, X[i][steps2+j], Y[i][steps2+j]) 
            YY[i][steps2+j] = -Bz_axial
            XX[i][steps2+j] = Bz_radial
            YY[i][steps2-j] = -Bz_axial
            XX[i][steps2-j] = -Bz_radial

    fig = plt.figure(facecolor='white', figsize=(17, 13))
    ax = plt.gca()
    speed = np.sqrt(XX*XX + YY*YY)
    

    strm = ax.streamplot(X, Y, XX, YY, color=speed, linewidth=1, density=[4, 4], arrowsize=1.5, cmap=plt.get_cmap('viridis_r'))
    fig.colorbar(strm.lines)

    plt.axis([-xmax, xmax, -ymax, ymax])
    fig.set_tight_layout(True)
    
    plt.show(block=False)    
    raw_input("tadaa!")

def draw_flux_contour():
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm 
    coil_r2 = 25.8 / 2 / 1000  # outer radios of the coil = 25.3/2 mm
    coil_h = 6.2 / 1000 # coil height is 6.2 mm

    # Parameters for prototype 3
    N = 3000  # number of turns
#    m_D = 12 / 1000 # diameter of the magnet = 9.525 mm
#    m_D = 9.525 / 1000 # diameter of the magnet = 9.525 mm
    m_D = 9.36 / 1000 # diameter of the magnet = 9.525 mm
    m_r = m_D/2
    m_h = 19.05 / 1000 # lenngth of the magnet = 19.05 mm
    m_Br = 1.2
    
    step = 0.001
    dd = np.arange(-0.03, 0.03+step, step)
    phi = np.zeros(dd.size)
    
    k_co = 0.55277
    d_co = 100e-6
    parts = 19

    steps  = 200
    steps2 = int(steps / 2)
    ymax = 50/1000
    xmax = 50/1000
    Y, X = np.mgrid[-ymax:ymax:200j, -xmax:xmax:200j]
    XX = np.zeros((steps,steps))
    YY = np.zeros((steps,steps))
    
    Br = 1.2

    for i in range(steps):
        for j in range(steps2):
            Bz_axial  = nasa_axial(Br, m_r, m_h/2, X[i][steps2+j], Y[i][steps2+j])
            Bz_radial = nasa_radial(Br, m_r, m_h/2, X[i][steps2+j], Y[i][steps2+j]) 
            YY[i][steps2+j] = -Bz_axial
            XX[i][steps2+j] = Bz_radial
            YY[i][steps2-j] = -Bz_axial
            XX[i][steps2-j] = -Bz_radial

    speed = np.sqrt(XX*XX + YY*YY)

    fig = plt.figure(facecolor='white', figsize=(17, 13))
    ax = plt.gca()    
    CS = plt.contour(X, Y, YY, 200, cmap=plt.get_cmap('viridis_r'))
#    plt.clabel(CS, fontsize=9, inline=1)
    plt.colorbar(CS).add_lines(CS, erase=True)
    plt.axis([-0.02, 0.02, -0.02, 0.02])
#    plt.axis([-0.03, 0.03, -0.03, 0.03])
    fig.set_tight_layout(True)
    plt.show(block=False)
    raw_input("tadaa!")
    plt.close()
    
    fig = plt.figure(facecolor='white', figsize=(17, 13))
    ax = plt.gca()    
    CS = plt.contour(X, Y, YY, 80, colors='k')
    plt.clabel(CS, fontsize=9, inline=1)
    plt.axis([-0.02, 0.02, -0.02, 0.02])
    fig.set_tight_layout(True)
    plt.show(block=False)
    raw_input("tadaa!")
    plt.close()
    
    

def test_flux_slice():
    m_D = 9.36 / 1000 # diameter of the magnet = 9.525 mm
    m_r = m_D/2
    m_h = 19.05 / 1000 # lenngth of the magnet = 19.05 mm
    m_Br = 1.2

    N = 3000  # number of turns    
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm 
    coil_r2 = 25.8 / 2 / 1000  # outer radios of the coil = 25.3/2 mm
    coil_h = 6.2 / 1000 # coil height is 6.2 mm
    k_co = 0.55277
    d_co = 100e-6
    parts = 7

#    z = 0; flux_linkage_nasa_slice(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, z, parts)

#    z = 1.00*m_h; flux_linkage_nasa_slice(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, z, parts)
#    z = 1.01*m_h; flux_linkage_nasa_slice(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, z, parts)
#    z = 0.99*m_h; flux_linkage_nasa_slice(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, z, parts)

#    z = 1.00*m_h/2; flux_linkage_nasa_slice(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, z, parts)
#    z = 1.01*m_h/2; flux_linkage_nasa_slice(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, z, parts)
#    z = 0.99*m_h/2; flux_linkage_nasa_slice(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, z, parts)
    
    print "Almost there!"
    
    r = 0.0; z =  0/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/1.14735-1)*100)
    r = 0.0; z =  9/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/0.686153-1)*100)
    r = 0.0; z = 10/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/0.548088-1)*100)
    r = 0.0; z = 20/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/0.04967-1)*100)
    
    r = 4.0/1000; z =  9/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/0.796188-1)*100)
    r = 5.0/1000; z =  9/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/-0.190783-1)*100)
    r = 4.0/1000; z = 10/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/0.451361-1)*100)
    r = 5.0/1000; z = 10/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/0.147657-1)*100)
    r = 8.0/1000; z =  9/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/-0.0277362-1)*100)
    r = 8.0/1000; z = 10/1000; Bz = Derby_axial(m_Br, m_r, m_h/2, r, z); print "r = %5.2f mm, z = %5.2f mm, Bz = % .6f T, diff = %6.2f %%" % (r*1000, z*1000, Bz, (Bz/-0.00605788-1)*100)


    
def draw_harvester_contours():
    calc_power(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, t0, resistivity)
    

def draw_power_contour(m_Br, h, coil_r2, gap, t0_per_h_coil, d_co, a, f):
    
#    coil_r2 = 6
#    h = 8.9
    
    k_co = 0.6
    parts = 25

    step = 0.01 / 1000
    min_r = 5.4 / 1000
    min_h = 2.4 / 1000
    h2 = 3.0 / 1000 # just to speed up the simulation temporarily
    r2 = 5.8 / 1000 # just to speed up the simulation temporarily
    x = np.arange(min_r, r2, step)
#    y = np.arange(min_h, h, step)
    y = np.arange(min_h, h2, step)
    X, Y = np.meshgrid(x*1000, y*1000)
    
    P = np.zeros_like(X)

    start = time.time()
    for i, coil_r1 in enumerate(x):
        for j, coil_h in enumerate(y):
            m_r = coil_r1 - gap
            t0 = t0_per_h_coil * coil_h
            m_h = h - coil_h + t0
            N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))
#            print "coil_r1 = %.1f mm, coil_h = %.1f mm, N = %d" % (coil_r1*1000 , coil_h*1000, N)
            P[j][i] = calc_power(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f) * 1000
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)  
    
    P_max_y, P_max_x = np.unravel_index(np.argmax(P), P.shape)
    P_max = P[P_max_y][P_max_x]
    xx = X[P_max_y][P_max_x]
    yy = Y[P_max_y][P_max_x]
    print "P_max_x = %d, p_max_y = %d, X[P_max_y][P_max_x] = %.2f mm, Y[P_max_y][P_max_x] = %.2f mm, P[P_max_y][P_max_x] = %.3f mW, t0 = %d %%" % (P_max_x, P_max_y, xx, yy, P_max, int(round(t0_per_h_coil*100)))

    fig = plt.figure(facecolor='white', figsize=(17, 13))
    ax = plt.gca()    
    
#    CS = plt.contour(X, Y, P, 20, cmap=plt.get_cmap('viridis_r'))
    CS = plt.contour(X, Y, P, 20, colors='k')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot(X[P_max_y][P_max_x], Y[P_max_y][P_max_x], 'o', label=r"$P_\mathrm{max}$")
    text_offset = 0.01
    ax.text(xx+text_offset, yy+text_offset, "%.2f mW" % (P_max), fontsize=15)
    ax.text(xx+text_offset, yy-2*text_offset, "(%.2f, %.2f)" % (xx, yy), fontsize=15)
#    plt.clabel(CS, fontsize=9, inline=1)
#    plt.colorbar(CS).add_lines(CS, erase=True)
#    plt.axis([-0.02, 0.02, -0.02, 0.02])
    plt.title(r"Output power for 1 $cm^3$ construction volume (mW)")
    plt.ylabel(r"$Coil_\mathrm{h}$")
    plt.xlabel(r"$Coil_\mathrm{r1}$")
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("power_contour.pdf")
    raw_input("tadaa!")
    plt.close()

def plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, print_P, d_co, N, norm):
    P_max_y, P_max_x = np.unravel_index(np.argmax(P), P.shape)
    P_max = P[P_max_y][P_max_x]
    P_xx  = X[P_max_y][P_max_x]
    P_yy  = Y[P_max_y][P_max_x]

    k_max_y, k_max_x = np.unravel_index(np.argmax(k), k.shape)
    k_max = k[k_max_y][k_max_x]
    k_xx  = X[k_max_y][k_max_x]
    k_yy  = Y[k_max_y][k_max_x]

    V_max_y, V_max_x = np.unravel_index(np.argmax(V), V.shape)
    V_max = V[V_max_y][V_max_x]
    V_xx  = X[V_max_y][V_max_x]
    V_yy  = Y[V_max_y][V_max_x]

    print "d_co = %d um, c_r = %.1f, c_h = %.1f, P_max = %.2f mW, V(P_max) = %.2f V, Rc = %d, Rl = %d, Z = %.2f, N = %d" % (int(d_co*1e6), P_xx, P_yy, P_max, V[P_max_y][P_max_x], int(round(Rc[P_max_y][P_max_x])), int(round(Rl[P_max_y][P_max_x])), Z[P_max_y][P_max_x], int(round(N[P_max_y][P_max_x]))) 

    if norm:
        P /= P_max
        
    fig = plt.figure(facecolor='white', figsize=(17, 13))

    legend_location = 3
    mark_size = 8
    contour_lines = 10
    plt.subplot(3,2,1)
    CS = plt.contour(X, Y, Z, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt = '%.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title("Max displacement Z (mm)")

    plt.subplot(3,2,2)
    CS = plt.contour(X, Y, Rc, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt = '%1.0f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title(r"Coil resistance ($\Omega$)")

    plt.subplot(3,2,3)
    CS = plt.contour(X, Y, Rl, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt = '%1.0f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title(r"Load resistance ($\Omega$)")

    plt.subplot(3,2,4)
    CS = plt.contour(X, Y, k, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt = '%0.1f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title("magnetic flux gradient (V/(m/s))")

    plt.subplot(3,2,5)
    CS = plt.contour(X, Y, V, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt = '%0.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)
    plt.title("Load voltage (V)")
    
    plt.subplot(3,2,6)
#    CS = plt.contour(X, Y, P, 20, cmap=plt.get_cmap('viridis_r'))
    CS = plt.contour(X, Y, P, contour_lines, colors='k')
    plt.clabel(CS, inline=1, fmt = '%0.2f', fontsize=10)
    plt.plot(k_xx, k_yy, '*', markersize=mark_size, label=r"$k_\mathrm{max}$")
    plt.plot(V_xx, V_yy, 'o', markersize=mark_size, label=r"$V_\mathrm{max}$")
    plt.plot(P_xx, P_yy, 'v', markersize=mark_size, label=r"$P_\mathrm{max}$")
    plt.legend(loc=legend_location)

    if print_P:
        ax = plt.gca()    
        text_offset = 2.0
        ax.text(P_xx, P_yy+text_offset, "%.2f mW" % (P_max), fontsize=15)
#        ax.text(P_xx+text_offset, P_yy-2*text_offset, "(%.2f, %.2f)" % (P_xx, P_yy), fontsize=15)
    plt.title(r"Output power (mW)")
    plt.ylabel(r"$Coil_\mathrm{h}$")
    plt.xlabel(r"$Coil_\mathrm{r1}$")
    
    
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig(outfile)
    raw_input("tadaa!")
    plt.close()


def draw_all_contours(outfile, m_Br, h, coil_r2, gap, t0_per_h_coil, d_co, a, f, two_coils, norm):
    
    k_co = 0.6
    parts = 25

    step = 0.2 / 1000
    min_r = 1.0 / 1000
    min_h = d_co
    x = np.arange(min_r, coil_r2, step)
    y = np.arange(min_h, h, step)
    X, Y = np.meshgrid(x*1000, y*1000)
    
    Z  = np.zeros_like(X)
    Rc = np.zeros_like(X)
    Rl = np.zeros_like(X)
    k  = np.zeros_like(X)
    V  = np.zeros_like(X)
    P  = np.zeros_like(X)
    N_arr  = np.zeros_like(X)

    start = time.time()
    for i, coil_r1 in enumerate(x):
        for j, coil_h in enumerate(y):
            m_r = coil_r1 - gap
            t0 = t0_per_h_coil * coil_h
            N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))
            N_arr[j,i] = N
            if two_coils:
                m_h = h - 2*coil_h + 2*t0
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
            else:
                m_h = h - coil_h + t0
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    Z *= 1000   # convert displacement to mm
    P *= 1000   # convert power to mW
    
    end = time.time()
    print "Elapsed time : %.2f seconds to calculate %d points (%.1f/s)" % (end-start, x.size*y.size, x.size*y.size/(end-start))  
    
    plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, False, d_co, N_arr, norm)


def draw_all_contours_magnet_fixed(outfile, m_Br, m_h, m_r, gap, max_r, max_h, t0_per_h_coil, d_co, a, f, two_coils, norm):
    
    k_co = 0.6
    parts = 25

    step = 0.2 / 1000
    coil_r1 = m_r + gap
    min_r = coil_r1 + d_co
    min_h = d_co
    x = np.arange(min_r, max_r, step)
    y = np.arange(min_h, max_h, step)
    X, Y = np.meshgrid(x*1000, y*1000)
    
    Z  = np.zeros_like(X)
    Rc = np.zeros_like(X)
    Rl = np.zeros_like(X)
    k  = np.zeros_like(X)
    V  = np.zeros_like(X)
    P  = np.zeros_like(X)
    N_arr  = np.zeros_like(X)

    start = time.time()
    for i, coil_r2 in enumerate(x):
        for j, coil_h in enumerate(y):
            t0 = t0_per_h_coil * coil_h
            N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))
            N_arr[j,i] = N
            if two_coils:
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
            else:
                Z[j][i], Rc[j][i], Rl[j][i], k[j][i], V[j][i], P[j][i] = calc_power_all(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    Z *= 1000   # convert displacement to mm
    P *= 1000   # convert power to mW
    
    end = time.time()
    print "Elapsed time : %.2f seconds to calculate %d points (%.1f/s)" % (end-start, x.size*y.size, x.size*y.size/(end-start))  
    
    plot_contours(outfile, X, Y, Z, Rc, Rl, k, V, P, True, d_co, N_arr, norm)


def main():
    
#    Foelsch_example()
#    test_Heuman_Lambda()
#    plot_bz()
#    draw_flux_lines()   
#    draw_flux_contour()
    
    Br = 1.2
    coil_r = 6.0e-3
    a = 20.0e-3
    r = 9.0e-3
    x = 30.0e-3
#    x1 = -10
#    x2 = 30
#    print "Br = %.1f T, coil_r = %.1f mm, r = %.1f mm, h_mag/2 = %.1f mm, x = %.1f mm" % (Br, coil_r*1000, r*1000, a*1000, x*1000)
    Bz_Foelsch1 = Foelsch1_axial(Br, coil_r, a, r, x)
    Bz_Foelsch2 = Foelsch2_axial(Br, coil_r, a, r, x)
    Bz_nasa = nasa_axial(Br, coil_r, a, r, x)
    Bz_derby = Derby_axial(Br, coil_r, a, r, x)

#    print "Bz_Foelsch1 = % 1.20f" % (Bz_Foelsch1)
#    print "Bz_Foelsch2 = % 1.20f" % (Bz_Foelsch2)
#    print "Bz_Nasa     = % 1.20f" % (Bz_nasa)
#    print "Bz_Derby    = % 1.20f" % (Bz_derby)

    test_bz_solvers()
#    test_flux_slice()

#    test_flux_linkage_N500()
#    test_flux_linkage()
#    test_flux_linkage2()
#    test_flux_linkage3()
#    test_flux_linkage5()


    m_Br = 1.1

    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm 
    coil_r2 = 25.8 / 2 / 1000  # outer radios of the coil = 25.3/2 mm
    coil_h = 6.2 / 1000 # coil height is 6.2 mm
    k_co = 0.55277
    d_co = 100e-6
#    m_D = 9.36 / 1000 # diameter of the magnet = 9.525 mm
    m_D = 9.525 / 1000 # diameter of the magnet = 9.525 mm
    m_r = m_D/2
    m_h = 19.05 / 1000 # lenngth of the magnet = 19.05 mm
    m_Br = 1.2
    t0 = 0.5
    resistivity = 2.176
#    calc_power_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, t0, resistivity)


    m_Br = 1.1
    coil_h = 2.53 / 1000
    coil_r2 = 6.0 / 1000
#    h = 0.01**3/(np.pi*coil_r2*coil_r2)
    h = 8.9 / 1000
    coil_r1 = 5.41 / 1000
    gap = 0.5 / 1000
    m_r = coil_r1 - gap
    m_h = 0.92 * h
    d_co = 40e-6
    k_co = 0.6
    density = 7.6
#    t0 = 0.75
    t0_per_h = 0.75
    resistivity = 13.6
    t0 = t0_per_h * coil_h
#    print "coil_r1 = %.3f mm, coil_r2 = %.3f mm, coil_h = %.3f mm, m_r = %.3f mm, m_h = %.4f mm, h = %.3f mm" % (coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, h*1000)

    Nz_float = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
    Nz = int(round(Nz_float))
    step = coil_h / Nz

#    for t0_per_h in np.arange(0.0, 1.0, 0.01):
#        t0 = t0_per_h * coil_h
#        calc_power_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, t0, resistivity)

    t0_per_h = 0.75
    t0 = t0_per_h * coil_h
#    print "\ncoil_r1 = %.3f mm, coil_r2 = %.3f mm, coil_h = %.3f mm, m_r = %.3f mm, m_h = %.4f mm, h = %.3f mm" % (coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, h*1000)
#    print "t0 = %.3f mm, coil_h = %.3f mm, t0/coil_h = %.2f %%" % (t0*1000, coil_h*1000, t0/coil_h  )
#    calc_power_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, t0, resistivity)


    m_Br = 1.1
    coil_h = 2.53 / 1000
    coil_r2 = 6.0 / 1000
    coil_r1 = 5.41 / 1000
    gap = 0.5 / 1000
    m_r = coil_r1 - gap
    d_co = 40e-6
    k_co = 0.6
#    t0 = 0.75
    t0_per_h = 0.75
    t0 = t0_per_h * coil_h

    h = 0.01 ** 3 / (np.pi * coil_r2 * coil_r2)
    m_h = 0.92 * h
#    print "\n\nh = %.3f mm, coil_r1 = %.3f mm, coil_r2 = %.3f mm, coil_h = %.3f mm, m_r = %.3f mm, m_h = %.4f mm, h = %.3f mm" % (h*1000, coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, h*1000)
#    calc_power_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, t0, resistivity)

    h = 8.9 / 1000
    m_h = 0.92 * h
#    print "\n\nh = %.3f mm, coil_r1 = %.3f mm, coil_r2 = %.3f mm, coil_h = %.3f mm, m_r = %.3f mm, m_h = %.4f mm, h = %.3f mm" % (h*1000, coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, h*1000)
#    calc_power_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, t0, resistivity)

    h = 9.0 / 1000
    m_h = 0.92 * h
#    print "\n\nh = %.3f mm, coil_r1 = %.3f mm, coil_r2 = %.3f mm, coil_h = %.3f mm, m_r = %.3f mm, m_h = %.4f mm, h = %.3f mm" % (h*1000, coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, h*1000)
#    calc_power_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, t0, resistivity)

    t0_per_h_coil = 0.75

#    h = 0.01**3/(np.pi*coil_r2*coil_r2)
    h = 8.9 / 1000
#    h = 9.0 / 1000
    a = 10.0
    f = 100.0
#    draw_all_contours("contour_test1.pdf", m_Br, h, coil_r2, gap, t0_per_h_coil, d_co, a, f, False, False) # two_coils = False, norm = False
#    draw_all_contours("contour_test2.pdf", m_Br, h, coil_r2, gap, t0_per_h_coil, d_co, a, f, True, False) # two_coils = False, norm = False

    ##########################################################
    # calculate optimal values for magnet used in drop tests #
    ##########################################################
    m_Br = 1.3
    m_h = 19.05 / 1000 # lenngth of the magnet = 19.05 mm
    m_D = 9.525 / 1000 # diameter of the magnet = 9.525 mm
    m_r = m_D/2
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm
    gap = coil_r1 - m_r
#    gap = 0.5 / 1000
    print "gap = %.2f mm" % (gap*1000)
    max_r = 30.0 / 1000
    max_h = 30.0 / 1000
    t0_per_h_coil = 0.75
    d_co = 200e-6
#    print "d_co = %d um" % (int(d_co*1e6))
    a = 10.0
    f = 100.0
#    draw_all_contours_magnet_fixed("m_fixed_coils1_norm.pdf", m_Br, m_h, m_r, gap, max_r, max_h, t0_per_h_coil, d_co, a, f, False, True)   # two_coils = False, norm = True
#    draw_all_contours_magnet_fixed("m_fixed_coils2_norm.pdf", m_Br, m_h, m_r, gap, max_r, max_h, t0_per_h_coil, d_co, a, f, True, True)    # two_coils = True, norm = False

if __name__ == "__main__":
    main()

