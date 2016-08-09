from __future__ import division
from scipy.special import ellipk, ellipe, ellipkinc, ellipeinc
from sympy import elliptic_k, elliptic_e, elliptic_pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from scipy import special
import numpy as np
from numba import jit

tol = 1e-5

@jit
def cel(kc, p, c, s):
    if kc == 0:
#        print "cel: return np.nan"
        return np.nan
    errtol = 0.000001
#    k = np.abs(kc)
    k = abs(kc)
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
#    while np.abs(g-k) > g*errtol:
    while abs(g-k) > g*errtol:
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
        
    z1 = z + b
    m1 = 4*a*r / (z1*z1 + (a+r)*(a+r))

    z2 = z - b
    m2 = 4*a*r / (z2*z2 + (a+r)*(a+r))

    if (a-r) == 0:
        phi1 = np.pi/2
        phi2 = np.pi/2
        BZ = Br / 4 * (  z1/np.pi*np.sqrt(m1/(a*r))*ellipk(m1) - (z2/np.pi*np.sqrt(m2/(a*r))*ellipk(m2) ) )
    else:
        phi1 = np.arctan(abs(z1/(a-r))) 
        phi2 = np.arctan(abs(z2/(a-r)))
        if z1 == 0:
            BZ = - Br / 4 * ( z2/np.pi*np.sqrt(m2/(a*r))*ellipk(m2) + (a-r)*z2/abs((a-r)*z2)*Heuman_Lambda(phi2, m2) )
        elif z2 == 0:
            BZ =   Br / 4 * ( z1/np.pi*np.sqrt(m1/(a*r))*ellipk(m1) + (a-r)*z1/abs((a-r)*z1)*Heuman_Lambda(phi1, m1) ) 
        else:
            BZ = Br / 4 \
             * (  z1/np.pi*np.sqrt(m1/(a*r))*ellipk(m1) + (a-r)*z1/abs((a-r)*z1)*Heuman_Lambda(phi1, m1) \
               - (z2/np.pi*np.sqrt(m2/(a*r))*ellipk(m2) + (a-r)*z2/abs((a-r)*z2)*Heuman_Lambda(phi2, m2)) )

    return BZ

def nasa_radial(Br, a, b, r, z):

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

#    if z1 == 0:
#        BZ = - Br / 4 * ( z2/np.pi*np.sqrt(m2/(a*r))*ellipk(m2) + (a-r)*z2/abs((a-r)*z2)*Heuman_Lambda(phi2, m2) )
#    elif z2 == 0:
#        BZ =   Br / 4 * ( z1/np.pi*np.sqrt(m1/(a*r))*ellipk(m1) + (a-r)*z1/abs((a-r)*z1)*Heuman_Lambda(phi1, m1) ) 
#    else:
    BZ = Br / np.pi * np.sqrt(a/r) \
        * (  (2-m1)/(2*np.sqrt(m1)) * ellipk(m1) - ellipe(m1)/np.sqrt(m1) \
         - ( (2-m2)/(2*np.sqrt(m2)) * ellipk(m2) - ellipe(m2)/np.sqrt(m2) ) )

    return BZ



@jit
def Foelsch1(Br, a, b, r, z):
    
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
    
    return BZ

@jit
def Derby(Br, a, b, r, z):
    """
    Derby calculates the magnetic flux density of a cylindrical permanent magnet
    
    Br = Residual Flux Density [T]
    a  = Coil radius [m]
    b  = magnet length / 2 [m]
    r  = radius from the z axis
    z  = z coordinate 
    """
    
    if ((abs(z-b)<tol) and (abs(r-a)<tol)) or ((abs(z+b)<tol) and (abs(r-a)<tol)):
#        print "manipulating r"
        r = 1.0001 * r
        
    z1    = z + b
    beta1 = z1/np.sqrt( z1*z1 + (r+a)*(r+a) )
    k1    = np.sqrt( ( z1*z1 + (a - r)*(a - r) ) / ( z1*z1 + (a + r)*(a + r) ) )

    z2    = z - b
    beta2 = z2/np.sqrt( z2*z2 + (r+a)*(r+a) )
    k2    = np.sqrt( ( z2*z2 + (a - r)*(a - r) ) / ( z2*z2 + (a + r)*(a + r) ) )

    gamma = (a - r) / (a + r)
    Bz = Br/np.pi * a/(a + r) * ( beta1*cel(k1,gamma*gamma,1,gamma) - beta2*cel(k2,gamma*gamma,1,gamma) )

    return Bz


#@profile
def flux_linkage_Foelsch2(Br, mag_h, mag_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts):#, parts2):
    Nz = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
    Nr  = 2 * (coil_r2 - coil_r1) / (d_co * np.sqrt(np.pi/k_co))
    dN = Nz * Nr / (parts * parts)

    phi = 0.0
    dz = coil_h / parts
    dz2 = dz / 2
    z = d - coil_h/2 + dz2
    for j in xrange(parts):
        r = 0
        dphi = 0.0

        dr = mag_r / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = Foelsch2(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dphi
            r += dr
        r += dr/100
            
        dr = (coil_r1 - mag_r) / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = Foelsch2(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dphi
            r += dr

        dr = (coil_r2 - coil_r1) / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = Foelsch2(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dN * dphi
            r += dr
        z += dz
    return phi
    
def flux_linkage_nasa_orig(Br, mag_h, mag_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts):#, parts2):
    Nz = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
    Nr  = 2 * (coil_r2 - coil_r1) / (d_co * np.sqrt(np.pi/k_co))
    dN = Nz * Nr / (parts * parts)

    
    phi = 0.0
    dz = coil_h / parts
    dz2 = dz / 2
    z = d - coil_h/2 + dz2
    for j in xrange(parts):
        r = 0
        dphi = 0.0

        dr = mag_r / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dphi
            r += dr
            
        dr = (coil_r1 - mag_r) / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dphi
            r += dr

        dr = (coil_r2 - coil_r1) / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dN * dphi
            r += dr
        z += dz
    return phi

def flux_linkage_nasa(Br, mag_h, mag_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts):#, parts2):
    Nz = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
    Nr  = 2 * (coil_r2 - coil_r1) / (d_co * np.sqrt(np.pi/k_co))
#    dN = Nz * Nr / (parts * parts)
    dN = Nz * Nr / (parts * parts * 3)

    
    phi = 0.0
#    dz = coil_h / parts
    dz = coil_h / (3*parts)
    dz2 = dz / 2
    z = d - coil_h/2 + dz2
#    for j in xrange(parts):
    for j in xrange(3*parts):
        r = 0
        dphi = 0.0

        dr = mag_r / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
#            phi += dphi
            r += dr
            
        dr = (coil_r1 - mag_r) / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
#            phi += dphi
            r += dr

        dr = (coil_r2 - coil_r1) / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dN * dphi
            r += dr
        z += dz
    return phi

def flux_linkage_nasa_all(Br, mag_h, mag_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts):#, parts2):
    Nz_float = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
    Nr_float = 2 * (coil_r2 - coil_r1) / (d_co * np.sqrt(np.pi/k_co))
    Nr = int(round(Nr_float))
    Nz = int(round(Nz_float))
#    dN = Nz * Nr / (parts * parts)
#    dN = Nz * Nr / (parts * parts * 3)
    dN = Nz_float/Nz * Nr_float/Nr
#    print "Nz = %d, Nz_float = %.3f, Nr = %d, Nr_float = %.3f, dN = %.3f, N_int = %d, N_float = %.2f, N = %.2f" % (Nz, Nz_float, Nr, Nr_float, dN, Nr*Nz, Nr_float*Nz_float, Nr*Nz*dN)
    
    phi = 0.0
#    dz = coil_h / parts
    dz = coil_h / (Nz)
    dz2 = dz / 2
    z = d - coil_h/2 + dz2
#    for j in xrange(parts):
    for j in xrange(Nz):
        r = 0
        dphi = 0.0

        dr = mag_r / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
#            phi += dphi
            r += dr
            
        dr = (coil_r1 - mag_r) / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
#            phi += dphi
            r += dr

        dr = (coil_r2 - coil_r1) / Nr
        dr2 = dr / 2
        for i in xrange(Nr):
            Bz = nasa(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dN * dphi
            r += dr
        z += dz
    return phi
    
@jit
def flux_linkage_Derby(Br, mag_h, mag_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts):
    Nz_float = 2 * coil_h / (d_co * np.sqrt(np.pi/k_co))
    Nr_float = 2 * (coil_r2 - coil_r1) / (d_co * np.sqrt(np.pi/k_co))
    Nr = int(round(Nr_float))
    Nz = int(round(Nz_float))
    dN = Nz_float/Nz * Nr_float/Nr
    
    phi = 0.0
    dz = coil_h / (Nz)
    dz2 = dz / 2
    z = d - coil_h/2 + dz2
    for j in xrange(Nz):
        r = 0
        dphi = 0.0

        dr = mag_r / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = Derby(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            r += dr
            
        dr = (coil_r1 - mag_r) / parts
        dr2 = dr / 2
        for i in xrange(parts):
            Bz = Derby(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            r += dr

        dr = (coil_r2 - coil_r1) / Nr
        dr2 = dr / 2
        for i in xrange(Nr):
            Bz = Derby(Br, mag_r, mag_h/2, r+dr2, z)
            dphi += Bz * np.pi * ( (r+dr)*(r+dr) - r*r )
            phi += dN * dphi
            r += dr
        z += dz
    return phi

def calc_flux_gradient(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, d):
    parts = 12

    k_co = np.pi * d_co * d_co * N / ( 4 * coil_h * ( coil_r2 - coil_r1 ) )
    
    Nz = int(round( 2.0 * coil_h / (d_co * np.sqrt(np.pi/k_co)) ))
#    Nz = 100

    step = coil_h / Nz
    
    y1 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d-step, parts)
    y2 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d+step, parts)
    k = ( y2 - y1 ) / (2*step)
    
    return k


def calc_power_orig(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, t0, resistivity):
    parts = 12
    
#    step = 0.0001
    
#    resistivity = 13.6
    
    Nz = int(round( 2.0 * coil_h / (d_co * np.sqrt(np.pi/k_co)) ))
    Nr = int(round( 2.0 * (coil_r2 - coil_r1) / (d_co * np.sqrt(np.pi/k_co)) ))
    N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))
    print "Nz = %d, Nr = %d, N = %d" % (Nz, Nr, N)
    step = coil_h / Nz
    d = -(m_h+coil_h)/2+t0-step; y1 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    d = -(m_h+coil_h)/2+t0+step; y2 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    k = ( y2 - y1 ) / (2*step)
    
#    print "Nz = %d, Nr = %d, N = %d" % (round(Nz), round(Nr), round(N))
    R_coil = N * np.pi * (coil_r2 + coil_r1) * resistivity
    dm = 0.1
    R_load = R_coil + k*k/dm
#    print "R_coil = %.2f Ohms, R_load = %.2f Ohms, k = %.5f" % (R_coil, R_load, k)
    
    de = k * k / ( R_coil + R_load )
    density = 7600.0
    m = m_h * np.pi * m_r * m_r * density
#    print "de = %.2f, m = %.2f g" % (de, m*1000)
    
    a = 10.0
    f = 100.0
    omega = 2 * np.pi * f
    Z = m * a / ((de+dm)*omega)
    speed = Z * omega
#    speed = a / ( omega * (2*(de+dm)/(2*m*omega)) )
    V = k * speed
    V_load = V * R_load / (R_coil + R_load)
    I = V / (R_coil + R_load)
    P = V_load * V_load / R_load
    d = -(m_h+coil_h)/2+t0
#    print "t0 = %.3f mm, coil_h = %.3f mm, t0/coil_h = %.2f %%" % (t0*1000, coil_h*1000, t0/coil_h  )
    step_percent = step/coil_h*100
    print "calc_power: d = %.3f mm, %.2f %%, %.3f m/s, Vall = %.3f V, V_load = %.4f V, P_load = %.3f mW, R_coil = %.2f Ohm, R_load = %d Ohm, k = %.3f V/(m/s)" % (d*1000, t0/coil_h, speed, V, V_load, P*1000, R_coil, round(R_load), k)
    return P

def calc_power(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f):
    parts = 12

    k_co = np.pi * d_co * d_co * N / ( 4 * coil_h * ( coil_r2 - coil_r1 ) )
    
    step = coil_h / 100
    d = -(m_h+coil_h)/2+t0-step; y1 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    d = -(m_h+coil_h)/2+t0+step; y2 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    k = ( y2 - y1 ) / (2*step)
    
#    print "Nz = %d, Nr = %d, N = %d" % (round(Nz), round(Nr), round(N))
    resistivity = 1.709e-8/(d_co*d_co*np.pi/4)
    R_coil = N * np.pi * (coil_r2 + coil_r1) * resistivity
    dm = 0.1
    R_load = R_coil + k*k/dm
#    print "R_coil = %.2f Ohms, R_load = %.2f Ohms, k = %.5f" % (R_coil, R_load, k)
    
    de = k * k / ( R_coil + R_load )
    density = 7600.0
    m = m_h * np.pi * m_r * m_r * density
#    print "de = %.2f, m = %.2f g" % (de, m*1000)
    
    omega = 2 * np.pi * f
    Z = m * a / ((de+dm)*omega)
    speed = Z * omega

    V = k * speed
    V_load = V * R_load / (R_coil + R_load)
    P = V_load * V_load / R_load

    return P

@jit
def calc_power_all(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f):
    parts = 12

    k_co = np.pi * d_co * d_co * N / ( 4 * coil_h * ( coil_r2 - coil_r1 ) )
    
    step = coil_h / 100
    d = -(m_h+coil_h)/2+t0-step; y1 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    d = -(m_h+coil_h)/2+t0+step; y2 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    k = ( y2 - y1 ) / (2*step)
    
#    print "Nz = %d, Nr = %d, N = %d" % (round(Nz), round(Nr), round(N))
    resistivity = 1.709e-8/(d_co*d_co*np.pi/4)
    R_coil = N * np.pi * (coil_r2 + coil_r1) * resistivity
    dm = 0.1
    R_load = R_coil + k*k/dm
#    print "R_coil = %.2f Ohms, R_load = %.2f Ohms, k = %.5f" % (R_coil, R_load, k)
    
    de = k * k / ( R_coil + R_load )
    density = 7600.0
    m = m_h * np.pi * m_r * m_r * density
#    print "de = %.2f, m = %.2f g" % (de, m*1000)
    
    omega = 2 * np.pi * f
    Z = m * a / ((de+dm)*omega)
    speed = Z * omega

    V = k * speed
    V_load = V * R_load / (R_coil + R_load)
    P = V_load * V_load / R_load

    return (Z, R_coil, R_load, k, V_load, P)

def calc_power_all_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f):
    parts = 12

    k_co = np.pi * d_co * d_co * N / ( 4 * coil_h * ( coil_r2 - coil_r1 ) )
    
    step = coil_h / 100
    d = -(m_h+coil_h)/2+t0-step; y1 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    d = -(m_h+coil_h)/2+t0+step; y2 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
#    k = ( y2 - y1 ) / (2*step)
    k = ( y2 - y1 ) / (step) # k is now doubled !!!!!!!!
    
#    print "Nz = %d, Nr = %d, N = %d" % (round(Nz), round(Nr), round(N))
    resistivity = 1.709e-8/(d_co*d_co*np.pi/4)
    R_coil = 2 * N * np.pi * (coil_r2 + coil_r1) * resistivity # resistance is now doubled !!!!!!
    dm = 0.1
    R_load = R_coil + k*k/dm
#    print "R_coil = %.2f Ohms, R_load = %.2f Ohms, k = %.5f" % (R_coil, R_load, k)
    
    de = k * k / ( R_coil + R_load )
    density = 7600.0
    m = m_h * np.pi * m_r * m_r * density
#    print "de = %.2f, m = %.2f g" % (de, m*1000)
    
    omega = 2 * np.pi * f
    Z = m * a / ((de+dm)*omega)
    speed = Z * omega

    V = k * speed
    V_load = V * R_load / (R_coil + R_load)
    P = V_load * V_load / R_load

    return (Z, R_coil, R_load, k, V_load, P)

def calc_power_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f):
    parts = 12

    k_co = np.pi * d_co * d_co * N / ( 4 * coil_h * ( coil_r2 - coil_r1 ) )
    
    step = coil_h / 100
    d = -(m_h+coil_h)/2+t0-step; y1 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    d = -(m_h+coil_h)/2+t0+step; y2 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
#    k = ( y2 - y1 ) / (2*step)
    k = ( y2 - y1 ) / (step) # k is now doubled !!!!!!!!
    
#    print "Nz = %d, Nr = %d, N = %d" % (round(Nz), round(Nr), round(N))
    resistivity = 1.709e-8/(d_co*d_co*np.pi/4)
    R_coil = 2 * N * np.pi * (coil_r2 + coil_r1) * resistivity # resistance is now doubled !!!!!!
    dm = 0.1
    R_load = R_coil + k*k/dm
#    print "R_coil = %.2f Ohms, R_load = %.2f Ohms, k = %.5f" % (R_coil, R_load, k)
    
    de = k * k / ( R_coil + R_load )
    density = 7600.0
    m = m_h * np.pi * m_r * m_r * density
#    print "de = %.2f, m = %.2f g" % (de, m*1000)
    
    omega = 2 * np.pi * f
    Z = m * a / ((de+dm)*omega)
    speed = Z * omega

    V = k * speed
    V_load = V * R_load / (R_coil + R_load)
    P = V_load * V_load / R_load

    return P


def calc_power_print(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f):
    parts = 12

    k_co = np.pi * d_co * d_co * N / ( 4 * coil_h * ( coil_r2 - coil_r1 ) )
    
    step = coil_h / 100
    d = -(m_h+coil_h)/2+t0-step; y1 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    d = -(m_h+coil_h)/2+t0+step; y2 = flux_linkage_Derby(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
    k = ( y2 - y1 ) / (2*step)
    
#    print "Nz = %d, Nr = %d, N = %d" % (round(Nz), round(Nr), round(N))
    resistivity = 1.709e-8/(d_co*d_co*np.pi/4)
    R_coil = N * np.pi * (coil_r2 + coil_r1) * resistivity
    dm = 0.1
    R_load = R_coil + k*k/dm
#    print "R_coil = %.2f Ohms, R_load = %.2f Ohms, k = %.5f" % (R_coil, R_load, k)
    
    de = k * k / ( R_coil + R_load )
    density = 7600.0
    m = m_h * np.pi * m_r * m_r * density
#    print "de = %.2f, m = %.2f g" % (de, m*1000)
    
    omega = 2 * np.pi * f
    Z = m * a / ((de+dm)*omega)
    speed = Z * omega

    V = k * speed
    V_load = V * R_load / (R_coil + R_load)
    P = V_load * V_load / R_load

    d = -(m_h+coil_h)/2+t0
    step_percent = step/coil_h*100
    print "calc_power: d = %.3f mm, %.2f %%, %.3f m/s, Vall = %.3f V, V_load = %.4f V, P_load = %.3f mW, R_coil = %.2f Ohm, R_load = %d Ohm, k = %.3f V/(m/s)" % (d*1000, t0/coil_h, speed, V, V_load, P*1000, R_coil, round(R_load), k)
    return P

def draw_flux_lines_coil(outfile, m_Br, m_r, m_h, coil_r1, coil_r2, coil_h, N, d_co, t0, P_max, two_coils):
    
    k_co = np.pi * d_co * d_co * N / ( 4 * coil_h * ( coil_r2 - coil_r1 ) )
    
    step = 0.001
    dd = np.arange(-0.03, 0.03+step, step)
    phi = np.zeros(dd.size)
    
    k_co = 0.55277
    d_co = 100e-6
    parts = 19

    steps  = 200
    steps2 = int(steps / 2)
    ymax = (m_h / 2 - t0 + coil_h) * 1.2
    xmax = coil_r2 * 1.2
    if ymax > xmax:
        xmax = ymax
    else:
        ymax = xmax

    Y, X = np.mgrid[-ymax:ymax:200j, -xmax:xmax:200j]
    B = np.zeros((steps,steps))
    
    Br = 1.2

    for i in range(steps):
        for j in range(steps2):
            Bz_axial  = nasa(Br, m_r, m_h/2, X[i][steps2+j], Y[i][steps2+j])
            B[i][steps2+j] = -Bz_axial
            B[i][steps2-j] = -Bz_axial

    fig = plt.figure(facecolor='white', figsize=(17, 15))
    ax = plt.gca()

    CS = plt.contour(X*1000, Y*1000, B, 50, colors='k')
    plt.clabel(CS, fontsize=9, inline=1)

    offset = 0.3e-3

    coil_h  *= 1000
    coil_r1 *= 1000
    coil_r2 *= 1000
    m_r     *= 1000
    m_h     *= 1000
    offset  *= 1000
    t0      *= 1000
    xmax    *= 950
    ymax    *= 950
    
    ax = plt.gca()
    ax.add_patch(patches.Rectangle((-coil_r2, m_h/2-t0), coil_r2-coil_r1, coil_h, facecolor='yellow', alpha=0.2))
    ax.text(-(coil_r2+coil_r1)/2, m_h/2-t0+0.8*coil_h, "Coil", ha = 'center', fontsize=24)
    ax.add_patch(patches.Rectangle((coil_r1, m_h/2-t0), coil_r2-coil_r1, coil_h, facecolor='yellow', alpha=0.2))

    ax.annotate("",[-coil_r2, m_h/2-t0-4*offset],[0, m_h/2-t0-4*offset],arrowprops=dict(lw=2, color='blue', arrowstyle='<->', mutation_scale = 20))
    ax.text(-coil_r2/2, m_h/2-t0-3*offset,"%.2f mm" % (coil_r2), ha = 'center', color = 'blue', fontsize=16)

    ax.annotate("",[-coil_r1, m_h/2-t0+5*offset],[0, m_h/2-t0+5*offset],arrowprops=dict(lw=2, color='blue', arrowstyle='<->'))
    ax.text(-coil_r1/2, m_h/2-t0+6*offset,"%.2f mm" % (coil_r1), ha = 'center', color = 'blue', fontsize=16)

    ax.annotate("",[-m_r, -m_h/2],[0, -m_h/2],arrowprops=dict(lw=2, color='blue', arrowstyle='<->'))
    ax.text(-(m_r)/2, -m_h/2+offset,"%.2f mm" % (m_r), ha = 'center', color = 'blue', fontsize=16)

    ax.annotate("",[0, -m_h/2],[0, m_h/2],arrowprops=dict(lw=2, color='blue', arrowstyle='<->'))
    ax.text(offset, 0,"%.2f mm" % (m_h), ha = 'left', color = 'blue', fontsize=16)

    ax.annotate("",[-coil_r2+offset, m_h/2-t0],[-coil_r2+offset, m_h/2-t0+coil_h],arrowprops=dict(lw=2, color='blue', arrowstyle='<->'))
    ax.text(-coil_r2+2*offset, m_h/2-t0+coil_h/2,"%.2f mm" % (coil_h), ha = 'left', color = 'blue', fontsize=16)

    ax.annotate("",[-coil_r1-offset, m_h/2-t0],[-coil_r1-offset, m_h/2],arrowprops=dict(lw=2, color='blue', arrowstyle='<->'))
    ax.text(-coil_r1-2*offset, m_h/2-t0/2,"%.2f mm" % (t0), ha = 'right', color = 'blue', fontsize=16)

    if two_coils:
        ax.add_patch(patches.Rectangle((-coil_r2, -m_h/2+t0-coil_h), coil_r2-coil_r1, coil_h, facecolor='yellow', alpha=0.2))
        ax.add_patch(patches.Rectangle((coil_r1, -m_h/2+t0-coil_h), coil_r2-coil_r1, coil_h, facecolor='yellow', alpha=0.2))

    title = r"$B_r = %.1f\,\mathrm{T},\,N = %d,\,d_\mathrm{co} = %d\,\mathrm{um},\,P_\mathrm{max} = %.1f\,\mathrm{mW}$" % (m_Br, N, int(round(d_co*1e6)), P_max)
    plt.title(title)
    plt.axis([-xmax, xmax, -ymax, ymax])
    plt.xlabel("(mm)")
    plt.ylabel("(mm)")
    fig.set_tight_layout(True)
    
    plt.show(block=False)    
    plt.savefig(outfile)
    raw_input("tadaa!")
