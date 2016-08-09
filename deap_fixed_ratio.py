from __future__ import division
import random
from deap import creator, base, tools, algorithms

import numpy as np
import matplotlib.pyplot as plt
import time

from magnet_flux import *

ratio_min = 0.05
ratio_max = 1.0 - ratio_min

V = 0.01**3 # 1 cm^3

xmax = 6.3
ymax = 3.0

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("coordinate", random.uniform, ratio_min, ratio_max)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.coordinate, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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

def evalOneMax(individual, coil_r2, h, gap, k_co, d_co, a, f, m_Br, printing):
    R_ratio  = individual[0]
    H_ratio  = individual[1]
    T_ratio  = individual[2]
#    T_ratio  = 0.75
    
    coil_r1 = R_ratio * coil_r2
    m_r     = coil_r1 - gap
    
#    coil_h = H_ratio * h / 2
    coil_h = H_ratio * h
    t0     = T_ratio * coil_h
#    m_h    = h - 2 * coil_h + 2 * t0
    m_h = h - coil_h + t0
    
    N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))

#    P = calc_power_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    P = calc_power(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    if printing:
        print "R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, coil_r1 = %.3f, coil_r2 = %.3f, coil_h = %.3f, m_h = %.3f, h = %.3f, N = %d, P = %.3f mW" % (R_ratio, H_ratio, T_ratio, coil_r1*1000, coil_r2*1000, coil_h*1000, m_h*1000, h*1000, N, P*1000)
    return P,


toolbox.register("evaluate", evalOneMax, coil_r2 = 6e-3, h = 8.9e-3, gap = 0.5e-3, k_co = 0.6, d_co = 40e-6 , a = 10.0, f = 50.0, m_Br = 1.1, printing = False)
toolbox.register("mate", tools.cxUniform, indpb=0.8)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.35, indpb=0.25)
#toolbox.register("select", tools.selRoulette)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max)) 

POPULATION = 10000

population = toolbox.population(n=POPULATION)

NGEN=20
for gen in range(NGEN):

    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    top100 = tools.selBest(population, k=POPULATION)
    top = np.array(top100)
    evalOneMax(individual = top100[0], coil_r2 = 6e-3, h = 8.9e-3, gap = 0.5e-3, k_co = 0.6, d_co = 40e-6 , a = 10.0, f = 50.0, m_Br = 1.1, printing = True)
    
#top10 = tools.selBest(population, k=10)
#print "\nBest individual after optimization:"
#print "x = %.2f, y = %.2f" % (top10[0][0]*xmax, top10[0][1]*ymax)
