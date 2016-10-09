from __future__ import division
import numpy as np
from numpy import sqrt
import matplotlib
import matplotlib.pyplot as plt

def calc_sector(inner_r, outer_r, start_angle, end_angle):
    t = np.linspace(start_angle, end_angle, 50)
    x = inner_r*np.cos(t[::-1])
    x = np.append(x, outer_r*np.cos(t))
    x = np.append(x, inner_r*np.cos(t[-1]))
    y = inner_r*np.sin(t[::-1])
    y = np.append(y, outer_r*np.sin(t))
    y = np.append(y, inner_r*np.sin(t[-1]))
    return x, y

def main():
    fig = plt.figure(facecolor='white', figsize=(10, 10))
    plt.axis([-9.1, 9.6, -9.1, 9.1])
#    plt.axvline(x=9)
#    plt.axhline(y=9)
    ax = plt.gca()

    Nphi = 8
    Nr = 3
    m1_r = 9
    m1_Rslice = (m1_r*m1_r)/Nr
    delta_phi = 2*np.pi/Nphi
    R_i_prev = 0.0
    for i in range(0, Nr):
        R_i = sqrt(R_i_prev*R_i_prev + m1_Rslice)
        ri = (R_i+R_i_prev)/2
        w = R_i - R_i_prev
        phi_prev = 0.0
        for j in range(1, Nphi+1):
            phi = j*delta_phi
            phi_center = phi + delta_phi/2
            print "R_i_prev = %.3f, R_i = %.3f, phi_prev = %.3f, phi = %.3f" % (R_i_prev, R_i, phi_prev, phi)
            x, y = calc_sector(R_i_prev, R_i, phi_prev, phi)
            plt.plot(x, y, color='k')
            xc = ri*np.cos(phi_center)
            yc = ri*np.sin(phi_center)
            plt.plot(xc, yc, '.', color='k')  # midpoints
            if i == 0:
                d = 2*R_i*np.sin(delta_phi/2)/(delta_phi/2*3)
            else:
                d = 2*(R_i*R_i + R_i*R_i_prev + R_i_prev*R_i_prev)*np.sin(delta_phi/2)/(delta_phi/2*3*(R_i+R_i_prev))
            xc = d*np.cos(phi_center)
            yc = d*np.sin(phi_center)
#            plt.plot(xc, yc, '.', color='b')  # centroids
            phi_prev = phi
        R_i_prev = R_i


#    ax.text(2.7, 1.7,r"$(r_{\mathrm{c},i},\, \phi_j)$",fontsize=16, color='b')  # centroids
    ax.text(1.6, 0.5,r"$(r_{\mathrm{m},i},\, \phi_j)$",fontsize=16, color='k')  # midpoints
    ax.text(0.5, -0.4,r"$R(1)=0$",fontsize=16, color='k')
    ax.text(5.2, -0.4,r"$R(2)$",fontsize=16, color='k')
    ax.text(7.4, -0.4,r"$R(3)$",fontsize=16, color='k')
    ax.text(9.0, -0.4,r"$R(4)$",fontsize=16, color='k')
    
    t = np.linspace(0.0, delta_phi, 100)
    xc = 0.5*np.cos(t)
    yc = 0.5*np.sin(t)
    plt.plot(xc, yc, '-', color='k')
    ax.text(0.5, 0.15,r"$\Delta\phi$",fontsize=16, color='k')
    
    fig.set_tight_layout(True)
    plt.axis('off')
    plt.show(block=False)
    plt.savefig('pics/surface_meshing.pdf')
    raw_input("Hit Enter...")
    

if __name__ == "__main__":
    main()