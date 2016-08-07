from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


"""

This file shows the third order polynomial fit to the data extracted from Figure 1 b) on an article [1]

[1] Mann, B.P. and Sims, N.D., 2009. Energy harvesting from the nonlinear oscillations of magnetic levitation. 
    Journal of Sound and Vibration, 319(1), pp.515-530.

"""

def main():
    F = np.array([5.10,  2.62,  1.17, 0.828, 0.690, 0.448, 0.345, 0.290, 0.269, 0.248, 0.234, 0.207, 0.194], dtype='float32')
    s = np.array([8.75, 16.25, 23.75, 26.43, 27.86, 30.89, 32.68, 34.11, 35.18, 35.57, 36.79, 37.68, 39.34], dtype='float32')
    s = np.divide(s, 1000) # convert mm to m

    p = np.polyfit(s, F, 3)
    pp = np.poly1d(p)
    xp = np.linspace(0.002,0.062, 61)

    fig = plt.figure(facecolor='white', figsize=(16, 9))
    plt.plot(s*1000, F, 'o', xp*1000, pp(xp), '-')
    plt.xlabel('Separation Distance (mm)')
    plt.ylabel('Repulsion Force (N)')
    fig.set_tight_layout(True)
    plt.savefig('pics/Mann_and_Sims_spring.pdf')
    plt.show()
    
if __name__ == "__main__":
    main()
