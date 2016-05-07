from __future__ import division
from scipy.special import ellipk, ellipe, ellipkinc, ellipeinc
from sympy import elliptic_k, elliptic_e, elliptic_pi
#from scipy import special
import numpy as np
from numba import jit

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import timeit
import time

def cel(kc, p, c, s):
    if kc == 0:
        return np.nan
    errtol = 0.000001
    k = np.abs(kc)
    pp = p
    cc = c
    ss = s
    em = 1.0
    if p > 0:
        pp = np.sqrt(p)
        ss = s / pp
    else:
        f = kc*kc
        q = 1.0 - f
        g = 1.0 - pp
        f = f - pp
        q = q*(ss - c*pp)
        pp = np.sqrt( f/g )
        cc = (c - ss)/g
        ss = -q/(g*g*pp) + cc*pp
    f = cc
    cc = cc + ss/pp
    g = k/pp
    ss = 2*(ss + f*g)
    pp = g + pp
    g = em
    em = k + em
    kk = k
    while np.abs(g-k) > g*errtol:
        k = 2*np.sqrt(kk)
        kk = k*em
        f = cc
        cc = cc + ss/pp
        g = kk/pp
        ss = 2*(ss + f*g)
        pp = g + pp
        g = em
        em = k + em
    return (np.pi/2*(ss + cc*em)/(em*(em + pp)))

def Heuman_Lambda(phi, m):
#    m=np.sin(m)*np.sin(m)
    if phi == np.pi/2:
        return 1.0
    if m == 1:
        m = 1 - 1e-9
    mdash = (1-m)
    
    K = ellipk(m)
    E = ellipe(m)
    incF = ellipkinc(phi, mdash)
    incE = ellipeinc(phi, mdash)
    
    HL = 2/np.pi * (E*incF + K*incE - K*incF)
    return HL

def nasa(Br, a, b, r, z):
    if ((z == b) and (r == a)) or ((z == -b) and (r == a)):
        r = 1.0001 * r
    if r==0:
        r=1e-100
#    if z == b:
#        z = z * 10001 / 10000
        
    z1 = z + b
    m1 = 4*a*r / (z1*z1 + (a+r)*(a+r))

    z2 = z - b
    m2 = 4*a*r / (z2*z2 + (a+r)*(a+r))

    if (a-r) == 0:
        phi1 = np.pi/2
        phi2 = np.pi/2
        BZ = Br / 4 * (  z1/np.pi*np.sqrt(m1/(a*r))*ellipk(m1) - (z2/np.pi*np.sqrt(m2/(a*r))*ellipk(m2) ) )
    else:
        phi1 = np.arctan(np.abs(z1/(a-r))) 
        phi2 = np.arctan(np.abs(z2/(a-r)))
        if z1 == 0:
            BZ = - Br / 4 * ( z2/np.pi*np.sqrt(m2/(a*r))*ellipk(m2) + (a-r)*z2/np.abs((a-r)*z2)*Heuman_Lambda(phi2, m2) )
        elif z2 == 0:
            BZ =   Br / 4 * ( z1/np.pi*np.sqrt(m1/(a*r))*ellipk(m1) + (a-r)*z1/np.abs((a-r)*z1)*Heuman_Lambda(phi1, m1) ) 
        else:
            BZ = Br / 4 \
             * (  z1/np.pi*np.sqrt(m1/(a*r))*ellipk(m1) + (a-r)*z1/np.abs((a-r)*z1)*Heuman_Lambda(phi1, m1) \
               - (z2/np.pi*np.sqrt(m2/(a*r))*ellipk(m2) + (a-r)*z2/np.abs((a-r)*z2)*Heuman_Lambda(phi2, m2)) )

    return BZ

def Foelsch1(Br, a, b, r, z):
#    if ((z == b) and (r == a)) or ((z == -b) and (r == a)):
    if a == r:
        r = 1.0001 * r
    z1 = z + b
    z2 = z - b
    Rr4 = 4*a*r
    Rrsquared = ((a+r)*(a+r))
    n = Rr4 / Rrsquared
    
    
    beta1 = Rrsquared / ( Rrsquared + z1*z1 )
    beta2 = Rrsquared / ( Rrsquared + z2*z2 )    
    m1 = n * beta1
    m2 = n * beta2
    
    sqrt1n = np.sqrt(1-n)
    
    if r <= a:
        A1 = (float(elliptic_k(m1)) + float(elliptic_pi(n, m1))*sqrt1n) * np.sqrt(1-beta1)
        A2 = (float(elliptic_k(m2)) + float(elliptic_pi(n, m2))*sqrt1n) * np.sqrt(1-beta2)
    else:
        A1 = (float(elliptic_k(m1)) - float(elliptic_pi(n, m1))*sqrt1n) * np.sqrt(1-beta1)
        A2 = (float(elliptic_k(m2)) - float(elliptic_pi(n, m2))*sqrt1n) * np.sqrt(1-beta2)

    if (z >= -b) and (z <= b):
        BZ = Br * (A2 + A1) / (2 * np.pi)
    elif z < -b:
        BZ = Br * (A2 - A1) / (2 * np.pi)
    else:
        BZ = Br * (A1 - A2) / (2 * np.pi)

#    print "Legendre:  z = %.4f, r = %.4f, A1 = % 1.5f, A2 = % 1.5f, B1 = % 1.5f, B2 = % 1.5f, Bz = % 1.5f" % (x, r, A1, A2, B1, B2, BZ)
    return BZ
    
def Foelsch2(Br, a, b, r, z):
    if ((z == b) and (r == a)) or ((z == -b) and (r == a)):
        r = 1.0001 * r
    z1 = z + b
    z2 = z - b
    n = 4*a*r/(a+r)**2
    beta1 = (a+r)**2 / ( (a+r)**2 + z1*z1)
    beta2 = (a+r)**2 / ( (a+r)**2 + z2*z2)    
    m1 = n * beta1
    m2 = n * beta2
    K1 = ellipk(m1)
    E1 = ellipe(m1)
    K2 = ellipk(m2)
    E2 = ellipe(m2)
    
    if m1 == 1:
        sin2phi = 0
    else:
        sin2phi=(1-n)/(1-m1)
    phi1=np.arcsin(np.sqrt(sin2phi))
    sin2b1 = 1-m1
    b1 = np.arcsin(np.sqrt(sin2b1))
    Finc = ellipkinc(phi1, sin2b1)
    if Finc == np.inf:
#        print "z = %.5f, r = %.5f, Finc = inf, phi1 = %.5f, sin2b1 = %.5f" % (z, r, phi1, sin2b1)
        Finc = 10e20
    Einc = ellipeinc(phi1, sin2b1)
    A1 = np.pi/2 + K1*np.sqrt(1-beta1)*(1+np.sqrt(1-n))+Finc*(K1-E1)-K1*Einc
    B1 = 2*K1*np.sqrt(1-beta1) - A1
    
    if m2 == 1:
        sin2phi = 0
    else:
        sin2phi=(1-n)/(1-m2)
    phi2=np.arcsin(np.sqrt(sin2phi))
    sin2b2 = 1-m2
    b2 = np.arcsin(np.sqrt(sin2b2))
    Einc = ellipeinc(phi2, sin2b2)
#    if (phi2 == np.pi/2) and (sin2):
    Finc = ellipkinc(phi2, sin2b2)
    if Finc == np.inf:
#        print "z = %.5f, r = %.5f, Finc = inf, phi2 = %.5f, sin2b2 = %.5f" % (z, r, phi2, sin2b2)
        Finc = 10e20
    A2 = np.pi/2 + K2*np.sqrt(1-beta2)*(1+np.sqrt(1-n))+Finc*(K2-E2)-K2*Einc
    B2 = 2*K2*np.sqrt(1-beta2) - A2

    if r <= a:
        if (z >= -b) and (z <= b):
            BZ = Br * (A2 + A1) / (2 * np.pi)
        elif z < -b:
            BZ = Br * (A2 - A1) / (2 * np.pi)
        else:
            BZ = Br * (A1 - A2) / (2 * np.pi)
#            BZ = Br * (A2 - A1) / (2 * np.pi)
    else:
        if (z >= -b) and (z <= b):
            BZ = Br * (B2 + B1) / (2 * np.pi)
        elif z < -b:
            BZ = Br * (B2 - B1) / (2 * np.pi)
        else:
            BZ = Br * (B1 - B2) / (2 * np.pi)
    
#    print "Legendre2: z = %.4f, r = %.4f, A1 = % 1.5f, A2 = % 1.5f, B1 = % 1.5f, B2 = % 1.5f, Bz = % 1.5f" % (z, r, A1, A2, B1, B2, BZ)
    
    return BZ

def Derby(Br, a, b, r, z):
    """
    Derby calculates the magnetic flux density of a cylindrical permanent magnet
    
    Br = Residual Flux Density [T]
    a  = Coil radius [m]
    b  = magnet length / 2 [m]
    r  = radius from the z axis
    z  = z coordinate 
    """
    
    if ((z == b) and (r == a)) or ((z == -b) and (r == a)):
        r = 1.0001 * r
#        r = 0.9999 * r
        
    z1    = z + b
    beta1 = z1/np.sqrt( z1*z1 + (r+a)*(r+a) )
    k1    = np.sqrt( ( z1*z1 + (a - r)*(a - r) ) / ( z1*z1 + (a + r)*(a + r) ) )

    z2    = z - b
    beta2 = z2/np.sqrt( z2*z2 + (r+a)*(r+a) )
    k2    = np.sqrt( ( z2*z2 + (a - r)*(a - r) ) / ( z2*z2 + (a + r)*(a + r) ) )

    gamma = (a - r) / (a + r)
    Bz = Br/np.pi * a/(a + r) * ( beta1*cel(k1,gamma*gamma,1,gamma) - beta2*cel(k2,gamma*gamma,1,gamma) )
    return Bz

def Foelsch_example():
    coil_r =  0.06 #  6 cm
    hmag   =  0.40 # 40 cm
    r      =  0.09 #  9 cm
    x1     = -0.10 # 10 cm
    x2     =  0.30 # 30 cm
    
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
    
    sin2phi2=(1-n)/(1-m2)
#    sin2phi2=0.04955 # This wrong value was given in Foelsch paper
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

    t_Foelsch1 = timeit.Timer(lambda: Foelsch1(Br, coil_r, hmag_per_2, r, z))
    t_Foelsch2 = timeit.Timer(lambda: Foelsch2(Br, coil_r, hmag_per_2, r, z))
    t_nasa = timeit.Timer(lambda: nasa(Br, coil_r, hmag_per_2, r, z))
    t_derby = timeit.Timer(lambda: Derby(Br, coil_r, hmag_per_2, r, z))


    no = 10000
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
    plt.title('Magnetic flux density calculation', fontsize=12)
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
    plt.title('Magnetic flux density calculation', fontsize=12)
    plt.xlabel('r')
    plt.ylabel('Time (s)', fontsize=10)
    plt.plot(zz, tt_Foelsch1, label="Foelsch1")
    plt.plot(zz, tt_Foelsch2, label="Foelsch2")
    plt.plot(zz, tt_nasa, label="Nasa")
    plt.plot(zz, tt_derby, label="Derby")
    plt.legend(loc=1)

    fig.set_tight_layout(True)
    plt.show(block=False)
#    plt.savefig(filename)
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
    for ind, z in enumerate(zz):
        z_Foelsch1[ind] = Foelsch1(Br, coil_r, hmag_per_2, r, z)
        z_Foelsch2[ind] = Foelsch2(Br, coil_r, hmag_per_2, r, z)
        z_nasa[ind]     = nasa(Br, coil_r, hmag_per_2, r, z)
        z_derby[ind]    = Derby(Br, coil_r, hmag_per_2, r, z)
        z_analytic[ind] = Br/2*((z+hmag_per_2)/np.sqrt((z+hmag_per_2)*(z+hmag_per_2)+coil_r*coil_r)-(z-hmag_per_2)/np.sqrt((z-hmag_per_2)*(z-hmag_per_2)+coil_r*coil_r))

    plt.plot(zz, z_Foelsch1, label="Foelsch1")
    plt.plot(zz, z_Foelsch2, label="Foelsch2")
    plt.plot(zz, z_derby, label = "Derby")
    plt.plot(zz, z_analytic, label = "Analytic")
    plt.plot(zz, z_nasa, label = "Nasa")
#    plt.xscale('log')
#    plt.yscale('log')
    plt.legend(loc=1)
    plt.show(block=False)
    
    Foelsch1_err = max(z_Foelsch1 - z_analytic)
    Foelsch2_err = max(z_Foelsch2 - z_analytic)
    derby_err = max(z_derby - z_analytic)
    nasa_err = max(z_nasa - z_analytic)
    
    print "Foelsch1_err = %e Foelsch2_err = %e, Derby_err = %e, Nasa_err = %e" % (Foelsch1_err, Foelsch2_err, derby_err, nasa_err)
    
    raw_input("Next radial sweep")
    z = hmag_per_2 * 9 / 10
    print "z = %.5f" % (z)
    plt.close()
    
    zz = np.arange(0.0, 0.02, 0.0001)    
    z_Foelsch1 = np.zeros(zz.size)
    z_Foelsch2 = np.zeros(zz.size)
    z_nasa = np.zeros(zz.size)
    z_derby = np.zeros(zz.size)
    for ind, r in enumerate(zz):
        z_Foelsch1[ind] = Foelsch1(Br, coil_r, hmag_per_2, r, z)
        z_Foelsch2[ind] = Foelsch2(Br, coil_r, hmag_per_2, r, z)
        z_nasa[ind]     = nasa(Br, coil_r, hmag_per_2, r, z)
        z_derby[ind]    = Derby(Br, coil_r, hmag_per_2, r, z)

    plt.plot(zz, z_Foelsch1, label="Foelsch1")
    plt.plot(zz, z_Foelsch2, label="Foelsch2")
    plt.plot(zz, z_derby, label = "Derby")
    plt.plot(zz, z_nasa, label = "Nasa")
#    plt.xscale('log')
#    plt.yscale('log')
    plt.legend(loc=1)
    plt.show(block=False)    
    
    raw_input("end of plotting")

def main():
#    mpmath.mp.dps = 5; 
#    mpmath.mp.pretty = True
    
    Foelsch_example()
#    test_Heuman_Lambda()
    plot_bz()
    
    
    Br = 1.2
    coil_r = 6.0e-3
    a = 20.0e-3
    r = 9.0e-3
    x = 30.0e-3
#    x1 = -10
#    x2 = 30
    print "Br = %.1f T, coil_r = %.1f mm, r = %.1f mm, h_mag/2 = %.1f mm, x = %.1f mm" % (Br, coil_r*1000, r*1000, a*1000, x*1000)
    Bz_Foelsch1 = Foelsch1(Br, coil_r, a, r, x)
    Bz_Foelsch2 = Foelsch1(Br, coil_r, a, r, x)
    Bz_nasa = nasa(Br, coil_r, a, r, x)
    Bz_derby = Derby(Br, coil_r, a, r, x)

    print "Bz_Foelsch1 = % 1.20f" % (Bz_Foelsch1)
    print "Bz_Foelsch2 = % 1.20f" % (Bz_Foelsch2)
    print "Bz_Nasa     = % 1.20f" % (Bz_nasa)
    print "Bz_Derby    = % 1.20f" % (Bz_derby)
    
    test_bz_solvers()
    
    
    """
    b = 10
    k = 60

    t1 = timeit.Timer(lambda: legendre(Br, coil_r, a, r, x))
    t2 = timeit.Timer(lambda: nasa(Br, coil_r, a, r, x))
    t3 = timeit.Timer(lambda: derby(Br, coil_r, a, r, x))

    accu = 10
    mpmath.mp.dps = accu;
    t_lege = t1.timeit(number=1000)
    t_nasa = t2.timeit(number=1000)
    t_derby = t3.timeit(number=1000)
    print "1000 loops for legendre() takes = %.6f s (accuracy = %2d digits)" % (t_lege, accu)
    print "1000 loops for nasa() takes = %.6f s" % (t_nasa)
    print "1000 loops for derby() takes = %.6f s" % (t_derby)
    print "nasa algorithm is %.1f times faster than legendre" % (t_lege/t_nasa)

    t1 = timeit.Timer(lambda: Heuman_Lambda_rad(b*np.pi/180, k*np.pi/180))
    t2 = timeit.Timer(lambda: Heuman_Lambda_special(b*np.pi/180, k*np.pi/180))
    t_mpmath = t1.timeit(number=1000)
    t_special = t2.timeit(number=1000)
    print "1000 loops for Heuman_Lambda_rad() takes = %.6f s (accuracy = %2d digits)" % (t_mpmath, accu)
    print "1000 loops for Heuman_Lambda_special() takes = %.6f s)" % (t_special)
    print "scipy.special functions are %.2f times faster than mpmath functions" % (t_mpmath/t_special)
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    bb = np.arange(0, np.pi/2, 0.1)
    xlen = len(bb)
    kk = np.arange(-1, 1, 0.1)
    ylen = len(kk)


    kkk, bbb = np.meshgrid(kk, bb)
    Z = np.empty(bbb.shape)
    for y in range(ylen):
        for x in range(xlen):
            Z[x, y] = Heuman_Lambda_special(bb[x], kk[y])
    surf = ax.plot_surface(kkk, bbb, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show(block=False)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for y in range(ylen):
        for x in range(xlen):
#            Z[x, y] = mpmath.ellippi(bb[x], kk[y])
            H = mpmath.ellippi(bb[x], kk[y])
    surf = ax.plot_surface(kkk, bbb, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show(block=False)
    raw_input("hahaa!")
    """

if __name__ == "__main__":
    main()

#K1 = mpmath.ellipk(m1)
#E1 = mpmath.ellipe(m1)
#PI1 = mpmath.ellippi(n, m1)
#K2 = mpmath.ellipk(m2)
#E2 = mpmath.ellipe(m2)
#PI2 = mpmath.ellippi(n, m2)

#print "n = %.2f, beta1 = %.5f, beta2 = %.5f" % (n, beta1, beta2)
#print "m1 = %.5f, m2 = %.5f" % (m1, m2)
#print "K1 = %.5f, E1 = %.5f, PI1 = %.5f" % (K1, E1, PI1)
#print "K2 = %.5f, E2 = %.5f, PI2 = %.5f" % (K2, E2, PI2)


# mpmath.ellipk(m, **kwargs)
#  Evaluates the complete elliptic integral of the first kind, K(m)
#  Note that the argument is the parameter m=k^2, not the modulus k which is sometimes used.
#
# mpmath.ellippi(*args)
# Called with two arguments n,mn,m, evaluates the complete elliptic integral of the third kind (n,m)=(n;/2,m).

#sqrt1n = np.sqrt(1-n)
#if r<=coil_r:
#    A1 = (mpmath.ellippk(m1) + mpmath.ellippi(n, m1)*sqrt1n) * np.sqrt(1-b1)
#    A2 = (mpmath.ellippk(m2) + mpmath.ellippi(n, m2)*sqrt1n) * np.sqrt(1-b2)
#else:
#    A1 = (mpmath.ellippk(m1) - mpmath.ellippi(n, m1)*sqrt1n) * np.sqrt(1-b1)
#    A2 = (mpmath.ellippk(m2) - mpmath.ellippi(n, m2)*sqrt1n) * np.sqrt(1-b2)
#
#if (x>x0) and (x<x+hmag):
#    Bx = Br0 * (A2 + A1)
#else:
#    BX = Br0 * (A2 - A1)
