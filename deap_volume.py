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

def evalOneMax(individual, volume, HR_max_ratio, gap, k_co, d_co, a, f, m_Br, two_coils, printing):
    HR_ratio = individual[0]
    R_ratio  = individual[1]
    H_ratio  = individual[2]
    T_ratio  = individual[3]

    r_o = (volume / (np.pi * HR_max_ratio * HR_ratio)) ** (1. / 3.)
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

    volumes = np.array([0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]) * 0.01 ** 3  # cm^3
    coils = [False, True]
    P1 = np.zeros(volumes.size)
    P2 = np.zeros(volumes.size)
    V1 = np.zeros(volumes.size)
    V2 = np.zeros(volumes.size)
    N1 = np.zeros(volumes.size)
    N2 = np.zeros(volumes.size)
    Rc1 = np.zeros(volumes.size)
    Rc2 = np.zeros(volumes.size)
    HR_ratio1 = np.zeros(volumes.size)
    HR_ratio2 = np.zeros(volumes.size)
    HR1 = np.zeros(volumes.size)
    HR2 = np.zeros(volumes.size)
    R_ratio1 = np.zeros(volumes.size)
    R_ratio2 = np.zeros(volumes.size)
    H_ratio1 = np.zeros(volumes.size)
    H_ratio2 = np.zeros(volumes.size)
    T_ratio1 = np.zeros(volumes.size)
    T_ratio2 = np.zeros(volumes.size)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()

    toolbox.register("individual", initES, creator.Individual, creator.Strategy,
                     size=4, imin=ratio_min, imax=ratio_max, smin=smin, smax=0.2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
    toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max))

    for two_coils in coils:
        for i, vol in enumerate(volumes):
            toolbox.register("evaluate", evalOneMax, volume=vol, HR_max_ratio=HR_max_ratio,
                             gap=gap, k_co=k_co, d_co=d_co, a=10.0, f=50.0, m_Br=m_Br,
                             two_coils=two_coils, printing=False)

            pop = toolbox.population(n=MU)
            hof = tools.HallOfFame(1)

            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)

            fit_stats = tools.Statistics(lambda ind: ind.fitness.values)
            fit_stats.register("avg", np.mean)
            fit_stats.register("std", np.std)
            fit_stats.register("min", np.min)
            fit_stats.register("max", np.max)

            stats_HR = tools.Statistics(lambda ind: ind[0])
            stats_HR.register("avg", np.mean)
            stats_HR.register("min", np.min)
            stats_HR.register("max", np.max)
            stats_HR.register("std", np.std)

            stats_R = tools.Statistics(lambda ind: ind[1])
            stats_R.register("avg", np.mean)
            stats_R.register("min", np.min)
            stats_R.register("max", np.max)
            stats_R.register("std", np.std)

            stats_H = tools.Statistics(lambda ind: ind[2])
            stats_H.register("avg", np.mean)
            stats_H.register("min", np.min)
            stats_H.register("max", np.max)
            stats_H.register("std", np.std)

            stats_T = tools.Statistics(lambda ind: ind[3])
            stats_T.register("avg", np.mean)
            stats_T.register("min", np.min)
            stats_T.register("max", np.max)
            stats_T.register("std", np.std)

            mstats = tools.MultiStatistics(fitness=fit_stats, HR=stats_HR, R=stats_R, H=stats_H, T=stats_T)
            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)

            t_start = time.time()
            pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                cxpb=0.6, mutpb=0.3, ngen=NGEN, stats=mstats, halloffame=hof)
            t_stop = time.time()
            print "Elapsed time = %.2f seconds." % (t_stop - t_start)


            gen = np.array(logbook.select("gen"))

            fig, axes = plt.subplots(facecolor='white', nrows=2, ncols=1, figsize=(12, 6))

            HR_maxs = logbook.chapters["HR"].select("max")
            HR_avgs = logbook.chapters["HR"].select("avg")
            HR_mins = logbook.chapters["HR"].select("min")
            axes[0].plot(gen, HR_maxs, '-k', label="HR_ratio")
            axes[0].plot(gen, HR_avgs, '--k')
            axes[0].plot(gen, HR_mins, '-k')

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
            axes[1].legend(loc=5)

            fig.subplots_adjust(wspace=0.01, hspace=0.1, left=0.05, right=0.99, top=0.98, bottom=0.04)
            plt.show(block=False)
            if two_coils:
                plt.savefig("pics/fixed_volume_2coils_progress_%d_cm3.pdf" % (i))
            else:
                plt.savefig("pics/fixed_volume_1coil_progress_%d_cm3.pdf" % (i))
    #        raw_input("hit any key!")
            plt.close()

            hof_array = np.array(hof)

            HR_ratio = hof_array[0][0]
            R_ratio  = hof_array[0][1]
            H_ratio  = hof_array[0][2]
            T_ratio  = hof_array[0][3]
            print "HR_ratio = %.3f, R_ratio = %.3f, H_ratio = %.3f, T_ratio = %.3f" % (HR_ratio, R_ratio, H_ratio, T_ratio)

            P_max = hof[0].fitness.values[0] * 1000

            r_o    = (vol / (np.pi * HR_max_ratio * HR_ratio)) ** (1. / 3.)
            r_i    = R_ratio * r_o
            r_mag  = r_i - gap
            h      = vol / (np.pi * r_o * r_o)
            h_coil = H_ratio * h
            t0     = T_ratio * h_coil

            if two_coils:
                h_mag    = h - 2 * h_coil + 2 * t0
            else:
                h_mag    = h - h_coil + t0

            N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))

            if two_coils:
                h_mag = h - 2 * h_coil + 2 * t0
                outfile = "pics/fixed_volum_2coils_optimum_%d_cm3.pdf" % (i)
                Z, R_coil, R_load, k, V_load, P = calc_power_all_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
                P2[i]        = P_max
                HR_ratio2[i] = HR_ratio
                R_ratio2[i]  = R_ratio
                H_ratio2[i]  = H_ratio
                T_ratio2[i]  = T_ratio
                V2[i]        = V_load
                N2[i]        = N
                Rc2[i]       = R_coil
                HR2[i]       = h / r_o
            else:
                h_mag = h - h_coil + t0
                outfile = "pics/fixed_volum_1coil_optimum_%d_cm3.pdf" % (i)
                Z, R_coil, R_load, k, V_load, P = calc_power_all(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)
                P1[i]        = P_max
                HR_ratio1[i] = HR_ratio
                R_ratio1[i]  = R_ratio
                H_ratio1[i]  = H_ratio
                T_ratio1[i]  = T_ratio
                V1[i]        = V_load
                N1[i]        = N
                Rc1[i]       = R_coil
                HR1[i]       = h / r_o


            draw_flux_lines_coil(outfile, m_Br, r_mag, h_mag, r_i, r_o, h_coil, N, d_co, t0, P_max, two_coils, False, a, f)

    print "Volume; HR_ratio; R_ratio; H_ratio; T_ratio; P; V; N; "
    for two_coils in coils:
        if two_coils:
            print "Two coils:"
        else:
            print "One coil:"            
        for i, vol in enumerate(volumes):
            if two_coils:
                print "%4.1f & %5.3f & %5.3f & %5.3f & %5.3f & %3d & %7.2f & %5.2f & %4d & %5.3f" % (vol/(0.01 ** 3), HR_ratio2[i], R_ratio2[i], H_ratio2[i], T_ratio2[i], N2[i], P2[i], V2[i], int(Rc2[i]), HR2[i])
            else:
                print "%4.1f & %5.3f & %5.3f & %5.3f & %5.3f & %3d & %7.2f & %5.2f & %4d & %5.3f" % (vol/(0.01 ** 3), HR_ratio1[i], R_ratio1[i], H_ratio1[i], T_ratio1[i], N1[i], P1[i], V1[i], int(Rc1[i]), HR1[i])

    print "starting to plot ratios..."
    vols = volumes / (0.01 ** 3)
    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(vols, HR_ratio1, 'k--', label=r"One coil $HR_\mathrm{ratio}$")
    plt.plot(vols, HR_ratio2, 'k-', label=r"Two coils $HR_\mathrm{ratio}$")
    plt.plot(vols, R_ratio1, 'r--', label=r"One coil $R_\mathrm{ratio}$")
    plt.plot(vols, R_ratio2, 'r-', label=r"Two coils $R_\mathrm{ratio}$")
    plt.plot(vols, H_ratio1, 'g--', label=r"One coil $H_\mathrm{ratio}$")
    plt.plot(vols, H_ratio2, 'g-', label=r"Two coils $H_\mathrm{ratio}$")
    plt.plot(vols, T_ratio1, 'b--', label=r"One coil $T_\mathrm{ratio}$")
    plt.plot(vols, T_ratio2, 'b-', label=r"Two coils $T_\mathrm{ratio}$")
    plt.xlabel(r"$volume\,\,[\mathrm{cm^3}]$", fontsize='large')
#    plt.ylabel(r"$\textit{power ratio}$", fontsize='x-large')
    plt.legend(loc=5)
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/volume_ratios.pdf")

    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(vols, HR1, 'b', label=r"One coil $h/r_\mathrm{o}$")
    plt.plot(vols, HR2, 'r', label=r"Two coils $h/r_\mathrm{o}$")
    plt.xlabel(r"$volume\,\,[\mathrm{cm^3}]$", fontsize='large')
#    plt.ylabel(r"$\textit{power ratio}$", fontsize='x-large')
    plt.legend(loc=5)
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/volume_aspect_ratios.pdf")


    raw_input("tadaa")

if __name__ == "__main__":
    main()
