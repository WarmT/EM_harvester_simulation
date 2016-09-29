from __future__ import division
import random
from deap import creator, base, tools, algorithms

import array
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import time

from magnet_flux import *

ratio_min = 0.03
ratio_max = 0.99
smin = 0.2

V = 0.01**3  # 1 cm^3

#r_mag = 9.525e-3 / 2
#h_mag = 19.05e-3
#m_V = np.pi * r_mag * r_mag * h_mag
#print "V = %.e, m_V = %.e, m_V/V = %.2f" % (V, m_V, m_V / V)

R_max_ratio = 3.0
#two_coils = True
a = 10.0
f = 50.0



# Individual generator
def initES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


def checkBounds(min, max):
    def decorator(func):
        def wrappper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrappper
    return decorator


def evalOneMax(individual, r_mag, h_mag, R_max_ratio, gap, k_co, d_co, a, f, m_Br, two_coils, printing):
    R_ratio  = individual[0]
    H_ratio  = individual[1]
    T_ratio  = individual[2]

    r_i = r_mag + gap
    r_o = r_i * (1 + R_ratio * R_max_ratio)
    if r_o - r_i < d_co:
        r_o = r_i + d_co
    h_coil  = H_ratio * h_mag
    t0      = T_ratio * h_coil

    N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))

    if two_coils:
        P = calc_power_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
    else:
        P = calc_power(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)

    if printing:
        print "R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, r_i = %.2f, r_o = %.2f, h_coil = %.2f, r_mag = %.2f, h_mag = %.2f, r_mag/h_mag = %.2f, N = %d, P = %.2f mW" % (R_ratio, H_ratio, T_ratio, r_i*1000, r_o*1000, h_coil*1000, r_mag*1000, h_mag*1000, r_mag/h_mag, N, P*1000)
    return P,

def main():
    random.seed()
    NGEN = 50
    MU, LAMBDA = 500, 500

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()

    toolbox.register("individual", initES, creator.Individual, creator.Strategy,
                     size=3, imin=ratio_min, imax=ratio_max, smin=smin, smax=0.2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
    toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max))

    coils = [False, True]

    r_oo = np.linspace(2e-3, 10e-3, 9)
    r_oo = r_oo[1:]

    P_ref = np.array([[0.0042, 0.0818, 0.6579, 3.1497, 10.8825, 30.2167, 71.9352, 152.8329],
                      [0.0070, 0.1257, 0.9410, 4.2365, 13.9104, 37.0865, 85.4832, 177.0028]])

    h_mag_ref = np.array([[3.9958, 5.3953, 6.8575, 8.3931, 9.9379, 11.5034, 13.0736, 14.6348],
                          [4.4206, 5.8986, 7.3746, 8.8514, 10.3294, 11.8073, 13.2843, 14.7622]]) * 1e-3

    r_mag_ref = np.array([[2.2059, 3.1460, 4.0945, 5.0453, 6.0445, 7.0441, 8.0425, 9.0450],
                          [2.2062, 3.1479, 4.0977, 5.0986, 6.0968, 7.0981, 8.1168, 9.1413]]) * 1e-3

    P = np.zeros(P_ref.shape)

    d_co = 50e-6
    k_co = 0.756 * 0.907  # k_co for d_co = 50 um
    m_Br = 1.1
    gap = 0.5e-3

    for j, two_coils in enumerate(coils):
        for i in xrange(P_ref.shape[1]):
            h_mag = h_mag_ref[j][i]
            r_mag = r_mag_ref[j][i]
#            print "j = %d, i = %d" % (j, i)
#            print "h_mag = %.4f, r_mag = %.4f" % (h_mag * 1000, r_mag * 1000)
            toolbox.register("evaluate", evalOneMax, r_mag=r_mag, h_mag=h_mag, R_max_ratio=R_max_ratio,
                             gap=gap, k_co=k_co, d_co=d_co, a=10.0, f=50.0, m_Br=m_Br,
                             two_coils=two_coils, printing=False)

            pop = toolbox.population(n=MU)
            hof = tools.HallOfFame(1)

            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)
            pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                                      cxpb=0.6, mutpb=0.3, ngen=NGEN, halloffame=hof)
            P_max = hof[0].fitness.values[0] * 1000
            P[j][i] = P_max

    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(r_oo * 1e3, P[1] / P_ref[1], 'r-', label="2 coils")
    plt.plot(r_oo * 1e3, P[0] / P_ref[0], 'b-', label="1 coil")
#    plt.plot(d_co_array, P2, label="Two Coils")
#    plt.ylabel(r"$[\mathrm{mW}]$ or $[\mathrm{V}]$", fontsize='large')
    plt.xlabel(r"$r_\mathrm{o}\,\,[\,\mathrm{mm}\,]$", fontsize='large')
    plt.legend(loc=1)
#    plt.yscale('log')
#    scale = plt.axis()
#    plt.axis([100, 250, scale[2], scale[3]])
#    plt.xticks(np.arange(100, 300, 50))
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/loop_magnet_fixed_rel.pdf")
    raw_input("tadaa")

    print "\nB_r = %.1f T, d_co = %d um, k_co = %.3f" % (m_Br, d_co * 1e6, k_co)

    print "d_co [um], k_co, P [mW], V [V], N, k, R_coil [Ohm], R_load [Ohm], r_i, r_o, h_coil, t0"
    for j, two_coils in enumerate(coils):
        print "%s" % ("Two_coils:" if two_coils else "One coil:  ")
        for i in xrange(P_ref.shape[1]):
            print "P_ratio = %.3f" % (P[j][i] / P_ref[j][i])


if __name__ == "__main__":
    main()
