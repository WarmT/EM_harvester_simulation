from __future__ import division
import mpmath
from scipy.special import ellipk, ellipe
from sympy import elliptic_k, elliptic_e, elliptic_pi, elliptic_f
from scipy import special
import numpy as np
from numba import jit
from scipy.interpolate import RectBivariateSpline

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import timeit
import time

def Heuman_Lambda_special(phi, m):
    if phi == np.pi/2:
        return 1.0
    if m == 1:
        m = 1 - 1e-9
    mdash = (1-m)
    
    K = special.ellipk(m)
    E = special.ellipe(m)
    incF = special.ellipkinc(phi, mdash)
    incE = special.ellipeinc(phi, mdash)
    
    HL = 2/np.pi * (E*incF + K*incE - K*incF)
    return HL

def Heuman_Lambda_sympy(phi, m):
    if phi == np.pi/2:
        return 1.0
    if m == 1:
        m = 1 - 1e-9
    mdash = (1-m)
    
    K = elliptic_k(m)
    E = elliptic_e(m)
    incF = elliptic_f(phi, mdash)
    incE = elliptic_e(phi, mdash)
    
    HL = 2/np.pi * (E*incF + K*incE - K*incF)
    return HL

def Heuman_Lambda_sympy2(phi, m):
    if phi == np.pi/2:
        return 1.0
    if m == 1:
        m = 1 - 1e-9
    mdash = (1-m)
    
    K = np.float64(elliptic_k(m))
    E = np.float64(elliptic_e(m))
    incF = np.float64(elliptic_f(phi, mdash))
    incE = np.float64(elliptic_e(phi, mdash))
    
    HL = 2/np.pi * (E*incF + K*incE - K*incF)
    return HL

def Heuman_Lambda_mixed(phi, m):
    if phi == np.pi/2:
        return 1.0
    if m == 1:
        m = 1 - 1e-9
    mdash = (1-m)
    
    K = elliptic_k(m)
    E = special.ellipe(m)
    incF = special.ellipkinc(phi, mdash)
    incE = special.ellipeinc(phi, mdash)
    
    HL = 2/np.pi * (E*incF + K*incE - K*incF)
    return HL

def Heuman_Lambda_mixed2(phi, m):
    if phi == np.pi/2:
        return 1.0
    if m == 1:
        m = 1 - 1e-9
    mdash = (1-m)
    
    K = float(elliptic_k(m))
    E = special.ellipe(m)
    incF = special.ellipkinc(phi, mdash)
    incE = special.ellipeinc(phi, mdash)
    
    HL = 2/np.pi * (E*incF + K*incE - K*incF)
    return HL

def eval_err(table, val):
    err = 1e-6
    if table == val:
        return "OK"
    elif (val-table) < err:
        return "OK"
    else:
        return "NOK"

def test_elliptic_functions():
    print "\nTesting elliptic integral functions\n"
    print "alpha_rad = alpha / 180 * pi"
    print "m = sin(alpha_rad)^2"
    print "K = special.ellipk(m) = Complete elliptic integral of first kind Legendre form"
    print "E = special.ellipe(m) = Complete elliptic integral of second kind Legendre form\n"

    print "+-------+---------+------------+----------------+------------+----------------+--------------------+"
    print "| alpha |    m    |     K      |       dK       |     E      |       dE       |      function      |"
    print "+-------+---------+------------+----------------+------------+----------------+--------------------+"
    
    alpha_test = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    K_table = [1.570796, 1.582843, 1.620026, 1.685750, 1.786769, 1.935581, 2.156516, 2.504550, 3.153385, np.inf]
    E_table = [1.570796, 1.558887, 1.523799, 1.467462, 1.393140, 1.305539, 1.211056, 1.118378, 1.040114, 1.0]

    for ind, alpha in enumerate(alpha_test):
        alpha_rad = alpha / 180 * np.pi
        m = np.sin(alpha_rad) ** 2
        
        K = special.ellipk(m)
        E = special.ellipe(m)
        if K == K_table[ind]:
            dK = 0.0
        else:
            dK = K - K_table[ind]
        dE = E - E_table[ind]
        print "|   %2d  | %.5f | %0.8f | % .2e (%s) | %.8f | % .2e (%s) | special.ellipk/e   |" % (alpha, m, K, dK, eval_err(K_table[ind], K), E, dE, eval_err(E_table[ind], E))

        K = mpmath.ellipk(m)
        E = mpmath.ellipe(m)
        if K == K_table[ind]:
            dK = 0.0
        else:
            dK = K - K_table[ind]
        dE = E - E_table[ind]
        print "|   %2d  | %.5f | %0.8f | % .2e (%s) | %.8f | % .2e (%s) | mpmath.ellipk/e    |" % (alpha, m, K, dK, eval_err(K_table[ind], K), E, dE, eval_err(E_table[ind], E))

        K = elliptic_k(m)
        E = elliptic_e(m)
        if K == K_table[ind]:
            dK = 0.0
        else:
            dK = K - K_table[ind]
        dE = E - E_table[ind]
        print "|   %2d  | %.5f | %0.8f | % .2e (%s) | %.8f | % .2e (%s) | sympy.elliptic_k/e |" % (alpha, m, K, dK, eval_err(K_table[ind], K), E, dE, eval_err(E_table[ind], E))

    print "+-------+---------+------------+----------------+------------+----------------+--------------------+\n"

    print "+-------+---------+-----------+------------------+"
    print "| alpha |    m    |   time    |     function     |"
    print "+-------+---------+-----------+------------------+"

    no = 1000
    alpha_test = np.linspace(0, 90, 900)
    special_k = np.zeros(len(alpha_test))
    mpmath_k  = np.zeros(len(alpha_test))
    sympy_k   = np.zeros(len(alpha_test))
    special_e = np.zeros(len(alpha_test))
    mpmath_e  = np.zeros(len(alpha_test))
    sympy_e   = np.zeros(len(alpha_test))
    
    t_special = timeit.Timer(lambda: special.ellipk(m))
    t_mpmath  = timeit.Timer(lambda: mpmath.ellipk(m))
    t_sympy   = timeit.Timer(lambda: elliptic_k(m))
    
    for ind, alpha in enumerate(alpha_test):
        alpha_rad = alpha / 180 * np.pi
        m = np.sin(alpha_rad) ** 2

        special_k[ind] = t_special.timeit(number=no) 
#        mpmath_k[ind]  = t_mpmath.timeit(number=no)
        sympy_k[ind]   = t_sympy.timeit(number=no)
        
#        print "|   %2d  | %.5f | %.5f s | special.ellipk   |" % (alpha, m, t_special)
#        print "|   %2d  | %.5f | %.5f s | mpmath.ellipk    |" % (alpha, m, t_mpmath)
#        print "|   %2d  | %.5f | %.5f s | sympy.elliptic_k |" % (alpha, m, t_sympy)
    print "+-------+---------+-----------+------------------+"

    t_special = timeit.Timer(lambda: special.ellipe(m))
    t_mpmath  = timeit.Timer(lambda: mpmath.ellipe(m))
    t_sympy   = timeit.Timer(lambda: elliptic_e(m))

    for ind, alpha in enumerate(alpha_test):
        alpha_rad = alpha / 180 * np.pi
        m = np.sin(alpha_rad) ** 2
        special_e[ind] = t_special.timeit(number=no)
#        mpmath_e[ind]  = t_mpmath.timeit(number=no)
        sympy_e[ind]   = t_sympy.timeit(number=no)
        
#        print "|   %2d  | %.5f | %.5f s | special.ellipe   |" % (alpha, m, t_special)
#        print "|   %2d  | %.5f | %.5f s | mpmath.ellipe    |" % (alpha, m, t_mpmath)
#        print "|   %2d  | %.5f | %.5f s | sympy.elliptic_e |" % (alpha, m, t_sympy)
    print "+-------+---------+-----------+------------------+\n"
    
    no = 1000
    
    steps = 20
    alpha_test2 = np.linspace(0, 90, steps)
    phi_test = np.linspace(0, 90, steps)
    
    spe_f = np.zeros(len(alpha_test2)*len(phi_test))
    sym_f = np.zeros(len(alpha_test2)*len(phi_test))
    
    t_special = timeit.Timer(lambda: special.ellipkinc(phi_rad, m))
    t_sympy   = timeit.Timer(lambda: elliptic_f(phi_rad, m))
    
    print "Starting to time Incomplete Elliptic Integrals of first kind"
    start = time.time()
    for ind_phi, phi in enumerate(phi_test):
        phi_rad = phi / 180 * np.pi
        for ind_alpha, alpha in enumerate(alpha_test2):
            alpha_rad = alpha / 180 * np.pi
            m = np.sin(alpha_rad)**2
            
            spe_f[ind_phi*steps+ind_alpha] = t_special.timeit(number=no) 
            sym_f[ind_phi*steps+ind_alpha]   = t_sympy.timeit(number=no)
            
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)
    
    spe_e = np.zeros(len(alpha_test2)*len(phi_test))
    sym_e = np.zeros(len(alpha_test2)*len(phi_test))
    sym_pi = np.zeros(len(alpha_test2)*len(phi_test))
    mpmath_pi = np.zeros(len(alpha_test2)*len(phi_test))
    
    t_special = timeit.Timer(lambda: special.ellipeinc(phi_rad, m))
    t_sympy   = timeit.Timer(lambda: elliptic_e(phi_rad, m))
    t_pi   = timeit.Timer(lambda: elliptic_pi(m_phi, m))
    t_mpmath   = timeit.Timer(lambda: mpmath.ellippi(m_phi, m))
    
    print "Starting to time Incomplete Elliptic Integrals of second kind"
    start = time.time()
    for ind_phi, phi in enumerate(phi_test):
        phi_rad = phi / 180 * np.pi
        m_phi = np.sin(phi_rad)**2
        if m_phi == 1.0:
            m_phi = 0.99
        for ind_alpha, alpha in enumerate(alpha_test2):
            alpha_rad = alpha / 180 * np.pi
            m = np.sin(alpha_rad)**2
            spe_e[ind_phi*steps+ind_alpha] = t_special.timeit(number=no) 
            sym_e[ind_phi*steps+ind_alpha] = t_sympy.timeit(number=no)
            sym_pi[ind_phi*steps+ind_alpha] = t_pi.timeit(number=no)
#            mpmath_pi[ind_phi*steps+ind_alpha] = t_mpmath.timeit(number=no)
            
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)

    fig = plt.figure(facecolor='white', figsize=(17, 9))

    plt.subplot(2,2,1)
    plt.title('Complete elliptic integral of first kind (Legendre form)', fontsize=12)
    plt.xlabel('Alpha (degree)')
    plt.ylabel('Time (s)', fontsize=10)
    plt.plot(alpha_test, special_k, label="special.ellipk")
#    plt.plot(alpha_test, t_mpmath_k, label="mpmath.ellipk")
    plt.plot(alpha_test, sympy_k, label="sympy.elliptic_k")
    plt.legend(loc=1)

    plt.subplot(2,2,2)
    plt.title('Incomplete elliptic integral of first kind (Legendre form)', fontsize=12)
#    plt.xlabel('Alpha (degree)')
    plt.ylabel('Time (s)', fontsize=10)
    plt.plot(spe_f, label="special.ellipkinc")
    plt.plot(sym_f, label="sympy.elliptic_f")
    plt.legend(loc=1)

    plt.subplot(2,2,3)
    plt.title('Complete elliptic integral of second kind (Legendre form)', fontsize=12)
    plt.xlabel('Alpha (degree)')
    plt.ylabel('Time (s)', fontsize=10)
    plt.plot(alpha_test, special_e, label="special.ellipe")
#    plt.plot(alpha_test, t_mpmath_e, label="mpmath.ellipe")
    plt.plot(alpha_test, sympy_e, label="sympy.elliptic_e")
    plt.legend(loc=1)

    plt.subplot(2,2,4)
    plt.title('Incomplete elliptic integral of second kind (Legendre form)', fontsize=12)
#    plt.xlabel('Alpha (degree)')
    plt.ylabel('Time (s)', fontsize=10)
    plt.plot(spe_e, label="special.ellipeinc")
    plt.plot(sym_e, label="sympy.elliptic_e")
    plt.plot(sym_pi, label="sympy.elliptic_pi")
#    plt.plot(mpmath_pi, label="mpmath.ellippi")
    plt.legend(loc=1)


    fig.set_tight_layout(True)
    plt.show(block=False)
#    plt.savefig(filename)
    raw_input("Hit enter")
    plt.close()


def test_elliptic_pi():
    
    phi_test = [0, 20, 30, 80, 89]
    alpha_test = [0, 30, 40, 90]
    pi_table = [ [0.0, 0.342020, 0.500000, 0.984808, 1.0], \
                 [0.0, 0.319707, 0.467777, 0.952751, 1.0], \
                 [0.0, 0.303869, 0.445330, 0.939042, 1.0], \
                 [0.0, 0.222222, 0.333333, 0.888889, 1.0] ]

    print "+-----+-------+---------+----------+----------+----------------+----------+"
    print "| phi | alpha |    m1   |    pi    | pi_table |      diff      | function |"
    print "+-----+-------+---------+----------+----------+----------------+----------+"
    for ind_phi, phi in enumerate(phi_test):
        phi_rad = phi / 180 * np.pi
        m1 = np.sin(phi_rad)**2
        for ind_alpha, alpha in enumerate(alpha_test):
            alpha_rad = alpha / 180 * np.pi
            m2 = np.sin(alpha_rad)**2
            pit = pi_table[ind_alpha][ind_phi]
            
            pii = elliptic_pi(m1, m2)
#            print "| %2d  |   %2d  | %.5f | %.6f | %.6f | % 1.2e (%s) | sympy    |" % (phi, alpha, m1, pii, pit, pit-pii, eval_err(pit, pii))
            print "| %2d  |   %2d  | %.5f | %.6f | %.6f | sympy    |" % (phi, alpha, m1, pii, pit,)


def test_Heuman_Lambda():
    print "\nTesting Heuman Lambda function\n"
    print "phi_rad = phi / 180 * pi"
    print "alpha_rad = alpha / 180 * pi"
    print "m = sin(alpha_rad)^2"
    print "HL = Heuman_Lambda(phi_rad, m)\n"
    
    phi_test = [0, 20, 30, 80, 90]
    alpha_test = [0, 30, 40, 90]
    HL_table = [ [0.0, 0.342020, 0.500000, 0.984808, 1.0], \
                 [0.0, 0.319707, 0.467777, 0.952751, 1.0], \
                 [0.0, 0.303869, 0.445330, 0.939042, 1.0], \
                 [0.0, 0.222222, 0.333333, 0.888889, 1.0] ]

    print "+-----+-------+---------+----------+----------+----------------+----------+"
    print "| phi | alpha |    m    |    HL    | HL_table |      diff      | function |"
    print "+-----+-------+---------+----------+----------+----------------+----------+"
    for ind_phi, phi in enumerate(phi_test):
        phi_rad = phi / 180 * np.pi
        for ind_alpha, alpha in enumerate(alpha_test):
            alpha_rad = alpha / 180 * np.pi
            m = np.sin(alpha_rad)**2
            HLt = HL_table[ind_alpha][ind_phi]
            
            HL = Heuman_Lambda_sympy(phi_rad, m)
            print "| %2d  |   %2d  | %.5f | %.6f | %.6f | % 1.2e (%s) | sympy    |" % (phi, alpha, m, HL, HLt, HLt-HL, eval_err(HLt, HL))
            
            HL = Heuman_Lambda_special(phi_rad, m)
            print "| %2d  |   %2d  | %.5f | %.6f | %.6f | % 1.2e (%s) | special  |" % (phi, alpha, m, HL, HLt, HLt-HL, eval_err(HLt, HL))
            
            HL = Heuman_Lambda_mixed(phi_rad, m)
            print "| %2d  |   %2d  | %.5f | %.6f | %.6f | % 1.2e (%s) | mixed    |" % (phi, alpha, m, HL, HLt, HLt-HL, eval_err(HLt, HL))
            
    print "+-----+-------+---------+----------+----------+----------------+----------+\n"
    
    no = 10000
    
    steps = 10
    alpha_test = np.linspace(0, 90, steps)
    phi_test = np.linspace(0, 90, steps)

    Z = np.zeros((len(phi_test), len(alpha_test)))
    for ind_phi, phi in enumerate(phi_test):
        phi_rad = phi / 180 * np.pi
        for ind_alpha, alpha in enumerate(alpha_test):
            alpha_rad = alpha / 180 * np.pi
            m = np.sin(alpha_rad)**2
            Z[ind_phi][ind_alpha] = Heuman_Lambda_special(phi_rad, m)
    
    y = phi_test/180*np.pi
    x = np.sin(alpha_test/180*np.pi)**2
    print "y.size = %d, y.shape = %s, Z.shape[0] = %s" % (y.size, y.shape, Z.shape[0])
    interp_spline = RectBivariateSpline(y, x, Z)
    
    spe  = np.zeros(len(alpha_test)*len(phi_test))
    sym  = np.zeros(len(alpha_test)*len(phi_test))
    mix  = np.zeros(len(alpha_test)*len(phi_test))
    mix2 = np.zeros(len(alpha_test)*len(phi_test))
    inte = np.zeros(len(alpha_test)*len(phi_test))
    
    t_special = timeit.Timer(lambda: Heuman_Lambda_special(phi_rad, m))
    t_sympy = timeit.Timer(lambda: Heuman_Lambda_sympy(phi_rad, m))
    t_mixed = timeit.Timer(lambda: Heuman_Lambda_mixed(phi_rad, m))
    t_mixed2 = timeit.Timer(lambda: Heuman_Lambda_mixed2(phi_rad, m))
    t_inte = timeit.Timer(lambda: interp_spline(m, phi_rad))
    
    start = time.time()
    for ind_phi, phi in enumerate(phi_test):
        phi_rad = phi / 180 * np.pi
        for ind_alpha, alpha in enumerate(alpha_test):
            alpha_rad = alpha / 180 * np.pi
            m = np.sin(alpha_rad)**2

            spe[ind_phi*steps+ind_alpha] = t_special.timeit(number=no) 
#            sym[ind_phi*steps+ind_alpha] = t_sympy.timeit(number=no)
#            mix[ind_phi*steps+ind_alpha] = t_mixed.timeit(number=no)
            mix2[ind_phi*steps+ind_alpha] = t_mixed2.timeit(number=no)
            inte[ind_phi*steps+ind_alpha] = t_inte.timeit(number=no)
            
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end-start)

    fig = plt.figure(facecolor='white', figsize=(17, 9))
    
    plt.title("Heuman's Lambda Function" , fontsize=12)
#    plt.xlabel('Alpha (degree)')
    plt.ylabel('Time (s)', fontsize=10)
    plt.plot(spe, label="special")
#    plt.plot(sym, label="sympy")
#    plt.plot(mix, label="mixed")
    plt.plot(mix2, label="mixed2")
    plt.plot(inte, label="Interp2d")
    plt.legend(loc=1)

    fig.set_tight_layout(True)
    plt.show(block=False)
#    plt.savefig(filename)
    raw_input("Hit enter")
    plt.close()
    
    
def main():

#    test_elliptic_functions()
#    test_elliptic_pi()
    test_Heuman_Lambda()
    

if __name__ == "__main__":
    main()
