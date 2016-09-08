from __future__ import division
import random
from deap import creator, base, tools, algorithms

import array
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import time

from magnet_flux import *

ratio_min = 0.04
ratio_max = 0.99

V = 0.01**3  # 1 cm^3

smin = 0.2

xmax = 6.3
ymax = 3.0

r_o = 6e-3
h = 8.9e-3
gap = 0.5e-3
k_co = 0.6
d_co = 40e-6
a = 10.0
f = 50.0
m_Br = 1.1
two_coils = False

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", array.array, typecode="d")


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
    R_ratio  = individual[0]
    H_ratio  = individual[1]
    T_ratio  = individual[2]
#    T_ratio  = 0.75

    r_i = R_ratio * r_o
    r_mag     = r_i - gap
    h_coil = H_ratio * h
    t0     = T_ratio * h_coil
    h_mag = h - h_coil + t0

    N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))

    if two_coils:
        P = calc_power_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
    else:
        P = calc_power(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
    if printing:

        print "R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, r_i = %.3f, r_o = %.3f, h_coil = %.3f, h_mag = %.3f, h = %.3f, N = %d, P = %.3f mW" % (R_ratio, H_ratio, T_ratio, r_i*1000, r_o*1000, h_coil*1000, h_mag*1000, h*1000, N, P*1000)
    return P,

toolbox = base.Toolbox()

toolbox.register("individual", initES, creator.Individual, creator.Strategy,
                 size=3, imin=ratio_min, imax=ratio_max, smin=smin, smax=0.2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalOneMax, r_o=r_o, h=h, gap=gap, k_co=k_co, d_co=d_co,
                 a=10.0, f=50.0, m_Br=m_Br, two_coils=two_coils, printing=False)

toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max))


def main():
    random.seed()
    NGEN = 50
    MU, LAMBDA = 100, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    fit_stats = tools.Statistics(lambda ind: ind.fitness.values)
    fit_stats.register("avg", np.mean)
    fit_stats.register("std", np.std)
    fit_stats.register("min", np.min)
    fit_stats.register("max", np.max)

    stats_R = tools.Statistics(lambda ind: ind[0])
    stats_R.register("avg", np.mean)
    stats_R.register("min", np.min)
    stats_R.register("max", np.max)
    stats_R.register("std", np.std)

    stats_H = tools.Statistics(lambda ind: ind[1])
    stats_H.register("avg", np.mean)
    stats_H.register("min", np.min)
    stats_H.register("max", np.max)
    stats_H.register("std", np.std)

    stats_T = tools.Statistics(lambda ind: ind[2])
    stats_T.register("avg", np.mean)
    stats_T.register("min", np.min)
    stats_T.register("max", np.max)
    stats_T.register("std", np.std)

    mstats = tools.MultiStatistics(fitness=fit_stats, R=stats_R, H=stats_H, T=stats_T)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    t_start = time.time()
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
        cxpb=0.6, mutpb=0.3, ngen=NGEN, stats=mstats, halloffame=hof)
    t_stop = time.time()
    print "Elapsed time = %.2f seconds." % (t_stop - t_start)

    gen = np.array(logbook.select("gen"))

    fig, axes = plt.subplots(facecolor='white', nrows=2, ncols=1, figsize=(12, 6))

    R_maxs = logbook.chapters["R"].select("max")
    R_avgs = logbook.chapters["R"].select("avg")
    R_mins = logbook.chapters["R"].select("min")
    axes[0].plot(gen, R_maxs, '-r', label="R_ratio")
    axes[0].plot(gen, R_avgs, '--r')
    axes[0].plot(gen, R_mins, '-r')

    H_maxs = logbook.chapters["H"].select("max")
    H_avgs = logbook.chapters["H"].select("avg")
    H_mins = logbook.chapters["H"].select("min")
    axes[0].plot(gen, H_maxs, '-g', label="H_ratio")
    axes[0].plot(gen, H_avgs, '--g')
    axes[0].plot(gen, H_mins, '-g')

    T_maxs = logbook.chapters["T"].select("max")
    T_avgs = logbook.chapters["T"].select("avg")
    T_mins = logbook.chapters["T"].select("min")
    axes[0].plot(gen, T_maxs, '-b', label="T_ratio")
    axes[0].plot(gen, T_avgs, '--b')
    axes[0].plot(gen, T_mins, '-b')
    axes[0].legend(loc=1)

    fit_maxs = np.array(logbook.chapters["fitness"].select("max")) * 1000
    fit_avgs = np.array(logbook.chapters["fitness"].select("avg")) * 1000
    fit_mins = np.array(logbook.chapters["fitness"].select("min")) * 1000
    axes[1].plot(gen, fit_maxs, '-b', label=r"$P_\mathrm{max}\,\,(\mathrm{mW})$")
    axes[1].plot(gen, fit_avgs, '--b')
    axes[1].plot(gen, fit_mins, '-b')
    axes[1].legend(loc=1)

    fig.subplots_adjust(wspace=0.01, hspace=0.1, left=0.03, right=0.99, top=0.98, bottom=0.04)
    plt.show(block=False)
    raw_input("hit any key!")
    if two_coils:
        plt.savefig("fixed_ratio_progress_es2.pdf")
    else:
        plt.savefig("fixed_ratio_progress_es.pdf")
    plt.close()

    hof_array = np.array(hof)

    R_ratio  = hof_array[0][0]
    H_ratio  = hof_array[0][1]
    T_ratio  = hof_array[0][2]

    r_i = R_ratio * r_o
    r_mag     = r_i - gap
    h_coil = H_ratio * h
    t0     = T_ratio * h_coil
    h_mag = h - h_coil + t0
    N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
    P_max = hof[0].fitness.values[0] * 1000

    if two_coils:
        outfile = "fixed_ratio_optimum_es2.pdf"
    else:
        outfile = "fixed_ratio_optimum_es.pdf"

    draw_flux_lines_coil(outfile, m_Br, r_mag, h_mag, r_i, r_o, h_coil, N, d_co, t0, P_max, two_coils, False, a, f)

    return pop, logbook, hof


if __name__ == "__main__":
    main()
