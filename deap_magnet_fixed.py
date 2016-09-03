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

m_r = 9.525e-3 / 2
m_h = 19.05e-3
m_V = np.pi * m_r * m_r * m_h
print "V = %.e, m_V = %.e, m_V/V = %.2f" % (V, m_V, m_V / V)

m_Br = 1.3
gap = 1.26e-3
R_max_ratio = 2.0
two_coils = True
k_co = 0.6
d_co = 100e-6


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


def evalOneMax(individual, m_r, m_h, R_max_ratio, gap, k_co, d_co, a, f, m_Br, two_coils, printing):
    R_ratio  = individual[0]
    H_ratio  = individual[1]
    T_ratio  = individual[2]

    coil_r1 = m_r + gap
    coil_r2 = coil_r1 * (1 + R_ratio * R_max_ratio)
    if coil_r2 - coil_r1 < d_co:
        coil_r2 = coil_r1 + d_co
    coil_h  = H_ratio * m_h
    t0      = T_ratio * coil_h

    N = int(round(4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co * d_co * np.pi)))

    if two_coils:
        P = calc_power_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    else:
        P = calc_power(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)

    if printing:
        print "R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, coil_r1 = %.2f, coil_r2 = %.2f, coil_h = %.2f, m_r = %.2f, m_h = %.2f, m_r/m_h = %.2f, N = %d, P = %.2f mW" % (R_ratio, H_ratio, T_ratio, coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, m_r/m_h, N, P*1000)
    return P,

toolbox = base.Toolbox()

toolbox.register("individual", initES, creator.Individual, creator.Strategy,
                 size=3, imin=ratio_min, imax=ratio_max, smin=smin, smax=0.2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalOneMax, m_r=m_r, m_h=m_h, R_max_ratio=R_max_ratio,
                 gap=gap, k_co=k_co, d_co=d_co, a=10.0, f=50.0, m_Br=m_Br,
                 two_coils=two_coils, printing=False)

toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max))

def main():
    random.seed()
    NGEN = 50
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

    fig, axes = plt.subplots(facecolor='white', nrows=4, ncols=1, figsize=(17, 9))

    R_maxs = logbook.chapters["R"].select("max")
    R_avgs = logbook.chapters["R"].select("avg")
    R_mins = logbook.chapters["R"].select("min")
    axes[0].plot(gen, R_maxs, '-b', label="max")
    axes[0].plot(gen, R_avgs, '--b', label="avg")
    axes[0].plot(gen, R_mins, '-b', label="min")
    axes[0].set_title("R_ratio")

    H_maxs = logbook.chapters["H"].select("max")
    H_avgs = logbook.chapters["H"].select("avg")
    H_mins = logbook.chapters["H"].select("min")
    axes[1].plot(gen, H_maxs, '-b', label="max")
    axes[1].plot(gen, H_avgs, '--b', label="avg")
    axes[1].plot(gen, H_mins, '-b', label="min")
    axes[1].set_title("H_ratio")

    T_maxs = logbook.chapters["T"].select("max")
    T_avgs = logbook.chapters["T"].select("avg")
    T_mins = logbook.chapters["T"].select("min")
    axes[2].plot(gen, T_maxs, '-b', label="max")
    axes[2].plot(gen, T_avgs, '--b', label="avg")
    axes[2].plot(gen, T_mins, '-b', label="min")
    axes[2].set_title("T_ratio")

    fit_maxs = np.array(logbook.chapters["fitness"].select("max")) * 1000
    fit_avgs = np.array(logbook.chapters["fitness"].select("avg")) * 1000
    fit_mins = np.array(logbook.chapters["fitness"].select("min")) * 1000
    axes[3].plot(gen, fit_maxs, '-b', label="max")
    axes[3].plot(gen, fit_avgs, '--b', label="avg")
    axes[3].plot(gen, fit_mins, '-b', label="min")
    axes[3].set_title(r"$P_\mathrm{max}\,\,(\mathrm{mW})$")

    fig.subplots_adjust(wspace=0.01, hspace=0.4, left=0.05, right=0.98, top=0.97, bottom=0.03)
    plt.show(block=False)
    raw_input("hit any key!")
    plt.savefig("optim_progress_es.pdf")
    plt.close()


    fit = np.zeros(len(pop))
    for i, p in enumerate(pop):
        fit[i] = p.fitness.values[0]
    print fit
    print "min = %.8f, max = %.8f, std = %.8f" % (np.min(fit), np.max(fit), np.std(fit))

    pop_array = np.array(pop)
    R_array =  pop_array[0, :]
    H_array =  pop_array[1, :]
    T_array =  pop_array[2, :]


    hof_array = np.array(hof)

    # fig, axes = plt.subplots(facecolor='white', figsize=(17, 9))
    R_ratio  = hof_array[0][0]
    H_ratio  = hof_array[0][1]
    T_ratio  = hof_array[0][2]

    coil_r1 = m_r + gap
    coil_r2 = coil_r1 * (1 + R_ratio * R_max_ratio)
    coil_h  = H_ratio * m_h
    t0      = T_ratio * coil_h
    N = int(round(4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co * d_co * np.pi)))
    P_max = hof[0].fitness.values[0] * 1000

    draw_flux_lines_coil("M_fixed_optimum_es.pdf", m_Br, m_r, m_h, coil_r1, coil_r2, coil_h, N, d_co, t0, P_max, two_coils)

    return pop, logbook, hof


if __name__ == "__main__":
    main()

