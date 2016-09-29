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

d_co = 50e-6
k_co = 0.756 * 0.907
r_o = 6e-3
h = 8.9e-3
gap = 0.5e-3
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
    NGEN = 100
    MU, LAMBDA = 500, 500

    r_o = 6e-3
    h = 8.9e-3
    ratio = h / r_o

    r_oo = np.linspace(2e-3, 10e-3, 9)
    P1 = np.zeros(r_oo.size)
    P2 = np.zeros(r_oo.size)
    V1 = np.zeros(r_oo.size)
    V2 = np.zeros(r_oo.size)
    N1 = np.zeros(r_oo.size)
    N2 = np.zeros(r_oo.size)
    Rc1 = np.zeros(r_oo.size)
    Rc2 = np.zeros(r_oo.size)
    R_ratio1 = np.zeros(r_oo.size)
    R_ratio2 = np.zeros(r_oo.size)
    H_ratio1 = np.zeros(r_oo.size)
    H_ratio2 = np.zeros(r_oo.size)
    T_ratio1 = np.zeros(r_oo.size)
    T_ratio2 = np.zeros(r_oo.size)
    M_var_p1 = np.zeros(r_oo.size)
    M_var_n1 = np.zeros(r_oo.size)
    M_var_p2 = np.zeros(r_oo.size)
    M_var_n2 = np.zeros(r_oo.size)

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

        R_ratio1[i]  = hof_array[0][0]
        H_ratio1[i]  = hof_array[0][1]
        T_ratio1[i]  = hof_array[0][2]

        print "R_ratio = %.3f, H_ratio = %.3f, T_ratio = %.3f" % (R_ratio1[i], H_ratio1[i], T_ratio1[i])

        r_i = R_ratio1[i] * r_o
        r_mag     = r_i - gap
        h_coil = H_ratio1[i] * h
        t0     = T_ratio1[i] * h_coil
        N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
        P_max = hof[0].fitness.values[0] * 1000

        h_mag = h - h_coil + t0

        Z, R_coil, R_load, k, V_load, P = calc_power_all(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)

        h_mag_per_h = h_mag / h
        t0_per_h_coil = t0 / h_coil
        print "One coil:  P_load = %.2f mW, V_load = %.2f V, r_i = %.2f mm, h_coil = %.2f mm, h_mag/h = %.3f, t0/h_coil = %.3f" % (P_max, V_load, r_i * 1000, h_coil * 1000, h_mag_per_h, t0_per_h_coil)
        print "P_max = %.4f mW, h_mag = %.4f mm, r_mag = %.4f mm" % (P_max, h_mag * 1000, r_mag * 1000)
        raw_input("daa")

        V1[i] = V_load
        N1[i] = N
        Rc1[i] = R_coil

        tolerance = 0.1 / 1000
        P_r_mag_p  = calc_power(m_Br, h_mag, r_mag + tolerance, h_coil, r_i, r_o, N, d_co, t0, a, f) * 1000  # convert to mW
        P_r_mag_n  = calc_power(m_Br, h_mag, r_mag - tolerance, h_coil, r_i, r_o, N, d_co, t0, a, f) * 1000
        M_var_p1[i] = (P_r_mag_p / P1[i] - 1) * 100
        M_var_n1[i] = (P_r_mag_n / P1[i] - 1) * 100
        print "P1[i] = %.6f, P_r_mag_p = %.6f, P_r_mag_n = %.6f" % (P1[i], P_r_mag_p, P_r_mag_n)
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

        R_ratio2[i]  = hof_array[0][0]
        H_ratio2[i]  = hof_array[0][1]
        T_ratio2[i]  = hof_array[0][2]

        print "R_ratio = %.3f, H_ratio = %.3f, T_ratio = %.3f" % (R_ratio2[i], H_ratio2[i], T_ratio2[i])

        r_i = R_ratio2[i] * r_o
        r_mag     = r_i - gap
        h_coil = H_ratio2[i] * h
        t0     = T_ratio2[i] * h_coil
        N = int(round(4.0 * h_coil * (r_o - r_i) * k_co / (d_co * d_co * np.pi)))
        P_max = hof[0].fitness.values[0] * 1000

        h_mag = h - 2 * h_coil + 2 * t0

        Z, R_coil, R_load, k, V_load, P = calc_power_all_two_coils(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f)

        h_mag_per_h = h_mag / h
        t0_per_h_coil = t0 / h_coil
        print "Two coils: P_load = %.2f mW, V_load = %.2f V, r_i = %.2f mm, h_coil = %.2f mm, h_mag/h = %.3f, t0/h_coil = %.3f" % (P_max, V_load, r_i * 1000, h_coil * 1000, h_mag_per_h, t0_per_h_coil)
        print "P_max = %.4f mW, h_mag = %.4f mm, r_mag = %.4f mm" % (P_max, h_mag * 1000, r_mag * 1000)
        raw_input("daa")

        V2[i] = V_load
        N2[i] = N
        Rc2[i] = R_coil

        tolerance = 0.1 / 1000
        P_r_mag_p  = calc_power_two_coils(m_Br, h_mag, r_mag + tolerance, h_coil, r_i, r_o, N, d_co, t0, a, f) * 1000  # convert to mW
        P_r_mag_n  = calc_power_two_coils(m_Br, h_mag, r_mag - tolerance, h_coil, r_i, r_o, N, d_co, t0, a, f) * 1000
        M_var_p2[i] = (P_r_mag_p / P2[i] - 1) * 100
        M_var_n2[i] = (P_r_mag_n / P2[i] - 1) * 100
        print "P1[i] = %.6f, P_r_mag_p = %.6f, P_r_mag_n = %.6f" % (P1[i], P_r_mag_p, P_r_mag_n)
        harvester_sensitivity(m_Br, h_mag, r_mag, h_coil, r_i, r_o, N, d_co, t0, a, f, two_coils, tolerance)

    #############
    # plotting  #
    #############

    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(r_oo * 1000, P2, 'r', label="Two Coils")
    plt.plot(r_oo * 1000, P1, 'b', label="One Coil")
    plt.ylabel(r"$P_\mathrm{load,max}\,\,[\mathrm{mW}]$", fontsize='large')
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$", fontsize='large')
    plt.legend(loc=2)
#    plt.yscale('log')
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/fixed_ratio_optim.pdf")
    raw_input("tadaa")

    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(r_oo * 1000, np.divide(P2, P1), label=r"$P_\mathrm{2\,coils}/P_\mathrm{1\,coil}$")
    plt.plot(r_oo * 1000, np.divide(V2, V1), label=r"$V_\mathrm{2\,coils}/V_\mathrm{1\,coil}$")
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$" , fontsize='large')
#    plt.ylabel(r"$\textit{power ratio}$", fontsize='x-large')
    plt.legend(loc=1)
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/fixed_ratio_optim_relative.pdf")
    raw_input("tadaa")


    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(r_oo * 1000, R_ratio1, 'r--', label=r"One coil $R_\mathrm{ratio}$")
    plt.plot(r_oo * 1000, R_ratio2, 'r-', label=r"Two coils $R_\mathrm{ratio}$")
    plt.plot(r_oo * 1000, H_ratio1, 'g--', label=r"One coil $H_\mathrm{ratio}$")
    plt.plot(r_oo * 1000, H_ratio2, 'g-', label=r"Two coils $H_\mathrm{ratio}$")
    plt.plot(r_oo * 1000, T_ratio1, 'b--', label=r"One coil $T_\mathrm{ratio}$")
    plt.plot(r_oo * 1000, T_ratio2, 'b-', label=r"Two coils $T_\mathrm{ratio}$")
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$", fontsize='large')
#    plt.ylabel(r"$\textit{power ratio}$", fontsize='x-large')
    plt.legend(loc=5)
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/fixed_ratio_ratios.pdf")
    raw_input("tadaa")

    fig, axes = plt.subplots(facecolor='white', figsize=(12, 6))
    plt.plot(r_oo * 1000, M_var_p1, 'b--', label=r"One coil $r_\mathrm{mag}+0.1\,\mathrm{mm}$")
    plt.plot(r_oo * 1000, M_var_n1, 'b-', label=r"One voil $r_\mathrm{mag}-0.1\,\mathrm{mm}$")
    plt.plot(r_oo * 1000, M_var_p2, 'r--', label=r"Two coils $r_\mathrm{mag}+0.1\,\mathrm{mm}$")
    plt.plot(r_oo * 1000, M_var_n2, 'r-', label=r"Two coils $r_\mathrm{mag}-0.1\,\mathrm{mm}$")
    plt.xlabel(r"$r_\mathrm{o}\,\,[\mathrm{mm}]$", fontsize='large')
    plt.ylabel(r"$\Delta P_\mathrm{max}\,\,[\,\%\,]$", fontsize='large')
    plt.legend(loc=1)
    fig.set_tight_layout(True)
    plt.show(block=False)
    plt.savefig("pics/fixed_ratio_magnet_variation.pdf")
    raw_input("tadaa")

    print "r_o [mm] & h [mm] & V [cm^3] & P_1 [mW] & V_1 [V] & N_1 & R_1 [Ohm] & P_2 [mW] & V_1 [V] & N_1 & R_1 [Ohm] "
    for i, r_o in enumerate(r_oo):
        h = ratio * r_o
        vol = np.pi * r_o * r_o * h / (0.01**3)  # construction volume in cm^3
        print "%5.2f & %5.2f & %5.2f & %10.6f & %5.2f & %3d & %3d & %10.6f & %5.2f & %3d & %3d" % (r_o * 1000, h * 1000, vol, P1[i], V1[i], N1[i], Rc1[i], P2[i], V2[i], N2[i], Rc2[i])

    print "\nd_co = %d um, k_co = %.3f" % (d_co*1e6, k_co)


if __name__ == "__main__":
    main()
