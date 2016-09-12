from __future__ import division
import random
from deap import creator, base, tools, algorithms

import array
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import time
import gc

from magnet_flux import *

ratio_min = 0.04
ratio_max = 0.99

V = 0.01**3  # 1 cm^3

smin = 0.2

xmax = 6.3
ymax = 3.0

k_co = 0.8
r_o = 6e-3
h = 8.9e-3
gap = 0.5e-3
d_co = 100e-6
a = 10.0
f = 50.0
m_Br = 1.1


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


def evalOneMax(individual, r_o, h, gap, k_co, d_co, a, f, m_Br, two_coils, printing):
    R_ratio = individual[0]
    H_ratio = individual[1]
    T_ratio = individual[2]
#    T_ratio  = 0.75

    r_i    = R_ratio * r_o
    r_mag  = r_i - gap
    if r_mag < d_co:
        r_mag = d_co
    h_coil = H_ratio * h
    t0     = T_ratio * h_coil
    N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
    if N == 0:
        N = 1

    if two_coils:
        h_mag = h - 2 * h_coil + 2 * t0
        if h_mag < 0:
            h_mag = d_co
        P = calc_power_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
    else:
        h_mag = h - h_coil + t0
        if h_mag < 0:
            h_mag = d_co
        P = calc_power(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)

    if printing:
        print "R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, r_i = %.3f, r_o = %.3f, h_coil = %.3f, h_mag = %.3f, h = %.3f, N = %d, P = %.3f mW" % (R_ratio, H_ratio, T_ratio, r_i*1000, r_o*1000, h_coil*1000, h_mag*1000, h*1000, N, P*1000)
    return P,

def main():

    random.seed()
    NGEN = 50
    MU, LAMBDA = 500, 500

    r_o = 6e-3
    h = 8.9e-3
    ratio = h / r_o

    r_oo = np.linspace(2e-3, 10e-3, 9)
    P1 = np.zeros(r_oo.size)
    P2 = np.zeros(r_oo.size)


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

    ############
    # one coil #
    ############
    two_coils = False
    toolbox.register("evaluate", evalOneMax, r_o=r_o, h=h, gap=gap, k_co=k_co, d_co=d_co,
                     a=10.0, f=50.0, m_Br=m_Br, two_coils=two_coils, printing=False)

    for i, r_o in enumerate(r_oo):
        h = ratio * r_o
        toolbox.unregister("evaluate")
        toolbox.register("evaluate", evalOneMax, r_o=r_o, h=h, gap=gap, k_co=k_co, d_co=d_co,
                         a=10.0, f=50.0, m_Br=m_Br, two_coils=two_coils, printing=False)

        pop = toolbox.population(n=MU)
        hof = tools.HallOfFame(1)
        fit_stats = tools.Statistics(lambda ind: ind.fitness.values)
        fit_stats.register("max", np.max)

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
            cxpb=0.6, mutpb=0.3, ngen=NGEN, stats=fit_stats, halloffame=hof)
 
        P1[i] = hof[0].fitness.values[0] * 1000

    hof_array = np.array(hof)

    R_ratio  = hof_array[0][0]
    H_ratio  = hof_array[0][1]
    T_ratio  = hof_array[0][2]

    r_i = R_ratio * r_o
    r_mag     = r_i - gap
    h_coil = H_ratio * h
    t0     = T_ratio * h_coil
    N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
    P_max = hof[0].fitness.values[0] * 1000

    h_mag = h - h_coil + t0
    outfile = "fixed_ratio_80_optimum_one_coil.pdf"

    draw_flux_lines_coil(outfile, m_Br, r_mag, h_mag, r_i, r_o, h_coil, N, d_co, t0, P_max, two_coils, False, a, f)

    Z, R_coil, R_load, k, V_load, P = calc_power_all(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)

    h_mag_per_h = h_mag / h
    t0_per_h_coil = t0 / h_coil
    print "One coil:  P_load = %.2f mW, V_load = %.2f V, r_i = %.2f mm, h_coil = %.2f mm, h_mag/h = %.3f, t0/h_coil = %.3f" % (P_max, V_load, r_i * 1000, h_coil * 1000, h_mag_per_h, t0_per_h_coil)

    tolerance = 0.1 / 1000
    harvester_sensitivity(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f, two_coils, tolerance)


    #############
    # two coils #
    #############
    two_coils = True
    toolbox.register("evaluate", evalOneMax, r_o=r_o, h=h, gap=gap, k_co=k_co, d_co=d_co,
                     a=10.0, f=50.0, m_Br=m_Br, two_coils=two_coils, printing=False)

    for i, r_o in enumerate(r_oo):
        h = ratio * r_o
        toolbox.unregister("evaluate")

        toolbox.register("evaluate", evalOneMax, r_o=r_o, h=h, gap=gap, k_co=k_co, d_co=d_co,
                         a=10.0, f=50.0, m_Br=m_Br, two_coils=two_coils, printing=False)
        
        pop = toolbox.population(n=MU)
        hof = tools.HallOfFame(1)
        fit_stats = tools.Statistics(lambda ind: ind.fitness.values)
        fit_stats.register("max", np.max)

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
            cxpb=0.6, mutpb=0.3, ngen=NGEN, stats=fit_stats, halloffame=hof)
 
        P2[i] = hof[0].fitness.values[0] * 1000

    hof_array = np.array(hof)

    R_ratio  = hof_array[0][0]
    H_ratio  = hof_array[0][1]
    T_ratio  = hof_array[0][2]

    r_i = R_ratio * r_o
    r_mag     = r_i - gap
    h_coil = H_ratio * h
    t0     = T_ratio * h_coil
    N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
    P_max = hof[0].fitness.values[0] * 1000

    h_mag = h - 2 * h_coil + 2 * t0
    outfile = "fixed_ratio_80_optimum_two_coils.pdf"

    draw_flux_lines_coil(outfile, m_Br, r_mag, h_mag, r_i, r_o, h_coil, N, d_co, t0, P_max, two_coils, False, a, f)

    Z, R_coil, R_load, k, V_load, P = calc_power_all_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)

    h_mag_per_h = h_mag / h
    t0_per_h_coil = t0 / h_coil
    print "Two coils: P_load = %.2f mW, V_load = %.2f V, r_i = %.2f mm, h_coil = %.2f mm, h_mag/h = %.3f, t0/h_coil = %.3f" % (P_max, V_load, r_i * 1000, h_coil * 1000, h_mag_per_h, t0_per_h_coil)

    tolerance = 0.1 / 1000
    harvester_sensitivity(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f, two_coils, tolerance)

    #############
    # plotting  #
    #############

    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(r_oo, P1, label="One Coil")
    plt.plot(r_oo, P2, label="Two Coils")
    plt.ylabel(r"$P_\mathrm{load,max}\,\,[\mathrm{mW}]$", fontsize='large')
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$", fontsize='large')
    plt.legend(loc=2)
#    plt.yscale('log')
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("fixed_ratio_optim.pdf")
    raw_input("tadaa")

    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(r_oo, np.divide(P2, P1), label=r"$P_\mathrm{2\,coils}/P_\mathrm{1\,coil}$")
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$" , fontsize='large')
#    plt.ylabel(r"$\textit{power ratio}$", fontsize='x-large')
    plt.legend(loc=1)
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("fixed_ratio_optim_relative.pdf")
    raw_input("tadaa")


if __name__ == "__main__":
    main()
