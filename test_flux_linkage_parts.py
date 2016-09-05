from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from magnet_flux import *
import time


def flux_linkage_parts():
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm
    coil_r2 = 25.8 / 2 / 1000  # outer radios of the coil = 25.3/2 mm
    coil_h = 6.2 / 1000  # coil height is 6.2 mm

    m_D = 9.525 / 1000  # diameter of the magnet = 9.525 mm
    m_r = m_D / 2
#    m_h = 19.05 / 1000  # lenngth of the magnet = 19.05 mm
    m_h = 1.05 / 1000  # lenngth of the magnet = 19.05 mm
    m_Br = 1.2

    k_co = 0.55277
    d_co = 100e-6

    d = 0.00 * m_h

    steps = 16
    FL_Derby = np.zeros(steps)

    start = time.time()
    ii = range(1, steps + 1)
    for i in ii:
        FL_Derby[i - 1] = flux_linkage_Derby_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, i)
    end = time.time()
    print "Elapsed time : %.2f seconds" % (end - start)

    plt.figure(facecolor='white', figsize=(12, 6))
    plt.plot(ii, FL_Derby, label="Derby (z = 0 mm)")
#    plt.legend(loc=4)
    plt.subplots_adjust(wspace=0.01, hspace=0.15, left=0.06, right=0.98, top=0.97, bottom=0.08)
    plt.xlabel("parts")
    plt.ylabel(r"Wb")
    plt.show(block=False)
    plt.savefig("flux_linkage_parts.pdf")
    raw_input("end of plotting")
    plt.close()


def test_flux_linkage_N500():
    coil_r1 = 12.05 / 2 / 1000   # inner radius of the coil = 12.05/2 mm
    coil_r2 = 16.3 / 2 / 1000    # outer radios of the coil = 25.3/2 mm
    coil_h = 12.1 / 1000         # coil height is 6.2 mm

    # Parameters for prototype 3
    N = 500  # number of turns
    m_D = 9.525 / 1000  # diameter of the magnet = 9.525 mm
    m_r = m_D / 2
    m_h = 19.05 / 1000  # lenngth of the magnet = 19.05 mm
    m_Br = 1.32

    d_co = 200e-6
    k_co = np.pi * d_co * d_co * N / (4 * coil_h * (coil_r2 - coil_r1))

    step = 0.0001
    step = coil_h / 100

    Nz_float = 2 * coil_h / (d_co * np.sqrt(np.pi / k_co))
    Nz = int(round(Nz_float))
    step = coil_h / (Nz)

    offset = 0.0
    dd = np.arange(-3 * m_h + offset, 3 * m_h + step + offset, step)
    FL_Derby_all = np.zeros(dd.size)

    parts = 25
    step_no = dd.size
    print "Number of steps: %d" % (step_no)
    start = time.time()
    t_start1 = time.time()
    for ind, d in enumerate(dd):
        tmp = flux_linkage_Derby_axial(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, k_co, d_co, d, parts)
        FL_Derby_all[ind] = tmp
    t_stop1 = time.time()
    print "flux_linkage_Derby_axial:      elapsed time = %.4f" % (t_stop1 - t_start1)

    end = time.time()
    print "Elapsed time : %.2f seconds (%.2f ms/step)" % (end - start, (end - start) / step_no * 1000)

    dz_flux_all = (FL_Derby_all[1:] - FL_Derby_all[:-1]) / step
    dz = dd[:-1] + step / 2

    fig, ax1 = plt.subplots(facecolor='white', figsize=(12, 6))
#    plt.title('Flux linkage')
    ax1.set_ylabel(r"$V/(m/s)$", color='r')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')
    ax1.plot(dz * 1000, dz_flux_all, 'r', label=r"$k_\mathrm{t}=\frac{\partial\varphi_\mathrm{m}}{\partial z}$")
    ax1.set_xlabel("z [mm]")
    ax1.legend(loc=2, fontsize='xx-large')

    plt.axvline(-m_h * 1000 / 2, 0, 1, color='black')
    plt.axvline(m_h * 1000 / 2, 0, 1, color='black')
    plt.axvline(0, 0, 1, color='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$Wb-t$", color='b')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')
#    ax2.plot(dd * 1000, FL_Derby_all, label="Magnetic flux linkage")
    ax2.plot(dd * 1000, FL_Derby_all, label=r"$\varphi_\mathrm{m}$")
    ax2.legend(loc=1, fontsize='x-large')

    scale = plt.axis()
    plt.axis([-30, 30, scale[2], scale[3]])

    plt.subplots_adjust(wspace=0.01, hspace=0.15, left=0.06, right=0.93, top=0.97, bottom=0.10)

    plt.show(block=False)
    raw_input("end of plotting")

    plt.savefig("flux_linkage_N500.pdf")
    plt.close()


def main():
    flux_linkage_parts()
    test_flux_linkage_N500()

if __name__ == "__main__":
    main()
