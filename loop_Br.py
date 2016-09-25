from __future__ import division
import random
from deap import creator, base, tools, algorithms

import numpy as np
import matplotlib.pyplot as plt
import time

from magnet_flux import *

import array
import multiprocessing


ratio_min = 0.02
ratio_max = 0.99

HR_max_ratio = 2.0
gap = 0.5e-3
d_co = 50e-6
k_co = 0.756 * 0.907
m_Br = 1.1
two_coils = True
PARAMS = 4
smin = 0.15
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

def evalOneMax(individual, volume, HR_ratio, gap, k_co, d_co, a, f, m_Br, two_coils, printing):
    R_ratio  = individual[0]
    H_ratio  = individual[1]
    T_ratio  = individual[2]

    r_o = (volume / (np.pi * HR_ratio)) ** (1. / 3.)
    r_i = R_ratio * r_o
    if r_o - r_i < d_co:
        r_i = r_o - d_co
    r_mag     = r_i - gap
    if r_mag <= 0:
        r_mag = r_i * 0.01

    h      = volume / (np.pi * r_o * r_o)
    h_coil = H_ratio * h

    if h_coil < d_co:
        h_coil = d_co

    t0     = T_ratio * h_coil

    if two_coils:
        h_mag    = h - 2 * h_coil + 2 * t0
    else:
        h_mag    = h - h_coil + t0

    if h_mag < d_co:
        h_mag = d_co

    N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
    if N == 0:
        N = 1

    if two_coils:
        P = calc_power_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
    else:
        P = calc_power(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)

    if printing:
        print "HR_ratio = %.2f, R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, r_i = %.2f, r_o = %.2f, h_coil = %.2f, r_mag = %.2f, h_mag = %.2f, r_mag/h_mag = %.2f, h = %.2f, N = %d, P = %.2f mW" % (HR_ratio, R_ratio, H_ratio, T_ratio, r_i*1000, r_o*1000, h_coil*1000, r_mag*1000, h_mag*1000, r_mag/h_mag, h*1000, N, P*1000)
    return P,


def main():
    nvolumes = 4

    random.seed()
    NGEN = 50
    MU, LAMBDA = 500, 500

    r_o = 6e-3
    h = 8.9e-3
    ratio = h / r_o

    br_array = np.linspace(0.8, 1.4, 13)
#    br_array = np.linspace(0.1, 0.9, 9)
    vol = 2 * 0.01 ** 3
    coils = [False, True]
    P1 = np.zeros(br_array.size)
    P2 = np.zeros(br_array.size)

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

    aspect = 1.0
    for two_coils in coils:
        for i, m_Br in enumerate(br_array):
            toolbox.register("evaluate", evalOneMax, volume=vol, HR_ratio=aspect,
                             gap=gap, k_co=k_co, d_co=d_co, a=10.0, f=50.0, m_Br=m_Br,
                             two_coils=two_coils, printing=False)

            pop = toolbox.population(n=MU)
            hof = tools.HallOfFame(1)

            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)

            pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                cxpb=0.6, mutpb=0.3, ngen=NGEN, halloffame=hof)

            if two_coils:
                P2[i] = hof[0].fitness.values[0] * 1000
            else:
                P1[i] = hof[0].fitness.values[0] * 1000


    print "B_r & P [mW]"
    for two_coils in coils:
        if two_coils:
            print "Two coils:"
        else:
            print "One coil:"            
        for i, m_Br in enumerate(br_array):
            if two_coils:
                print "%4.2f & %5.3f" % (m_Br, P2[i])
            else:
                print "%4.2f & %5.3f" % (m_Br, P1[i])

    print "\naspect = %.1f T, d_co = %d um, k_co = %.3f" % (aspect, d_co*1e6, k_co)

    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(br_array, P2, 'r', label=r"Two coils $P_\mathrm{load}$")
    plt.plot(br_array, P1, 'b', label=r"One coil $P_\mathrm{load}$")
    plt.xlabel(r"$B_\mathrm{r}$", fontsize='large')
    plt.ylabel(r"$P_\mathrm{load}\,\,[\,\mathrm{mW}\,]$", fontsize='large')
    plt.legend(loc=4)
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/loop_br.pdf")


    raw_input("tadaa")

if __name__ == "__main__":
    main()
