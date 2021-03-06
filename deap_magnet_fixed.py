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

toolbox = base.Toolbox()

toolbox.register("individual", initES, creator.Individual, creator.Strategy,
                 size=3, imin=ratio_min, imax=ratio_max, smin=smin, smax=0.2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalOneMax, r_mag=r_mag, h_mag=h_mag, R_max_ratio=R_max_ratio,
                 gap=gap, k_co=k_co, d_co=d_co, a=10.0, f=50.0, m_Br=m_Br,
                 two_coils=two_coils, printing=False)

toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max))


def main():
    random.seed()
    NGEN = 30
    MU, LAMBDA = 50, 100
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

    fig, axes = plt.subplots(facecolor='white', nrows=2, ncols=1, figsize=(17, 9))

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

    fig.subplots_adjust(wspace=0.01, hspace=0.1, left=0.02, right=0.99, top=0.98, bottom=0.03)
    plt.show(block=False)
    raw_input("hit any key!")
    if two_coils:
        plt.savefig("pics/M_optim_progress_two_coils.pdf")
    else:
        plt.savefig("pics/M_optim_progress_one_coils.pdf")
    plt.close()

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

    t0_per_h_coil = t0 / h_coil

    if two_coils:
        outfile = "pics/M_fixed_optimum_two_coils.pdf"
        Z, R_coil, R_load, k, V_load, P = calc_power_all_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
        print "Two coils: P_load = %.2f mW, V_load = %.2f V, r_i = %.2f, r_o = %.2f, h_coil = %.2f, r_mag = %.2f, h_mag = %.2f, t0/h_coil = %.3f, gap = %.2f, N = %d, R_load = %d" % (P_max, V_load, r_i * 1000, r_o * 1000, h_coil * 1000, r_mag * 1000, h_mag * 1000, t0_per_h_coil, gap * 1000, N, R_load)
    else:
        outfile = "pics/M_fixed_optimum_one_coil.pdf"
        Z, R_coil, R_load, k, V_load, P = calc_power_all(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
        print "One coil:  P_load = %.2f mW, V_load = %.2f V, r_i = %.2f, r_o = %.2f, h_coil = %.2f, r_mag = %.2f, h_mag = %.2f, t0/h_coil = %.3f, gap = %.2f, N = %d, R_load = %d" % (P_max, V_load, r_i * 1000, r_o * 1000, h_coil * 1000, r_mag * 1000, h_mag * 1000, t0_per_h_coil, gap * 1000, N, R_load)

    draw_flux_lines_coil(outfile, m_Br, r_mag, h_mag, r_i, r_o, h_coil, N, d_co, t0, P_max, two_coils, False, a, f)


    print "\nB_r = %.1f T, d_co = %d um, k_co = %.3f" % (m_Br, d_co*1e6, k_co)

#    return pop, logbook, hof


if __name__ == "__main__":
    main()
