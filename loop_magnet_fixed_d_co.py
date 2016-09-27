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

r_mag = 9.525e-3 / 2
h_mag = 19.05e-3
m_V = np.pi * r_mag * r_mag * h_mag
print "V = %.e, m_V = %.e, m_V/V = %.2f" % (V, m_V, m_V / V)

m_Br = 1.31
gap = 1.26e-3
R_max_ratio = 3.0
two_coils = True
d_co = 150e-6
k_co = 0.790 * 0.907  # k_co for d_co = 100 um
k_co = 0.812 * 0.907  # k_co for d_co = 100 um
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
    MU, LAMBDA = 100, 100

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
    S1_array   = np.array([0.790, 0.812, 0.826, 0.832])
    d_co_array = np.array([100e-6, 150e-6, 200e-6, 250e-6])
    P1 = np.zeros(d_co_array.size)
    P2 = np.zeros(d_co_array.size)
    V1 = np.zeros(d_co_array.size)
    V2 = np.zeros(d_co_array.size)

    for two_coils in coils:
        for i, d_co in enumerate(d_co_array):
            k_co = 0.907 * S1_array[i]

            toolbox.register("evaluate", evalOneMax, r_mag=r_mag, h_mag=h_mag, R_max_ratio=R_max_ratio,
                             gap=gap, k_co=k_co, d_co=d_co, a=10.0, f=50.0, m_Br=m_Br,
                             two_coils=two_coils, printing=False)

            pop = toolbox.population(n=MU)
            hof = tools.HallOfFame(1)

            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)

            pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                                      cxpb=0.6, mutpb=0.3, ngen=NGEN, halloffame=hof)

            hof_array = np.array(hof)

            R_ratio  = hof_array[0][0]
            H_ratio  = hof_array[0][1]
            T_ratio  = hof_array[0][2]

            r_i = r_mag + gap
            r_o = r_i * (1 + R_ratio * R_max_ratio)
            h_coil  = H_ratio * h_mag
            t0      = T_ratio * h_coil
            N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
            P_max = hof[0].fitness.values[0] * 1000

            if two_coils:
                P2[i] = P_max
                Z, R_coil, R_load, k, V_load, P = calc_power_all_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
                V2[i] = V_load
            else:
                P1[i] = P_max
                Z, R_coil, R_load, k, V_load, P = calc_power_all(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
                V1[i] = V_load


    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(d_co_array*1e6, P2, 'r-', label=r"$P_\mathrm{load,max}$ (2 coils)")
    plt.plot(d_co_array*1e6, P1, 'b-', label=r"$P_\mathrm{load,max}$ (1 coil)")
    plt.plot(d_co_array*1e6, V2, 'r--', label=r"$V_\mathrm{load}$ (2 coils)")
    plt.plot(d_co_array*1e6, V1, 'b--', label=r"$V_\mathrm{load}$ (1 coil)")
#    plt.plot(d_co_array, P2, label="Two Coils")
    plt.ylabel(r"$[\mathrm{mW}]$ or $[\mathrm{V}]$", fontsize='large')
    plt.xlabel(r"$d_\mathrm{co}\,\,[\mathrm{um}]$", fontsize='large')
    plt.legend(loc=1)
#    plt.yscale('log')
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("loop_magnet_fixed_d_co.pdf")
    raw_input("tadaa")


    print "\nB_r = %.1f T, d_co = %d um, k_co = %.3f" % (m_Br, d_co*1e6, k_co)

#    return pop, logbook, hof


if __name__ == "__main__":
    main()
