from __future__ import division
import random
from deap import creator, base, tools, algorithms

import numpy as np
import matplotlib.pyplot as plt
import time

from magnet_flux import *

ratio_min = 0.01
#ratio_max = 1.0 - ratio_min
ratio_max = 0.99

#m_V = np.pi * ((9.525e-3) ** 2) * 19.05e-3 / 4.
m_V = 0.01 ** 3

HR_max_ratio = 2.0
R_max_ratio = 4.0
k_co = 0.6
d_co = 100e-6
gap = 0.5e-3
m_Br = 1.3
two_coils = False

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("coordinate", random.uniform, ratio_min, ratio_max)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.coordinate, n=4)
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

def evalOneMax(individual, m_V, HR_max_ratio, R_max_ratio, gap, k_co, d_co, a, f, m_Br, two_coils, printing):
    HR_ratio = individual[0]
    R_ratio  = individual[1]
    H_ratio  = individual[2]
    T_ratio  = individual[3]
    
    m_r = ( m_V / ( np.pi * HR_max_ratio * HR_ratio ) ) ** (1./3.)
    m_h =   m_V / ( np.pi * m_r * m_r )
    
    coil_r1 = m_r + gap
    coil_r2 = d_co + coil_r1 * ( 1 + R_ratio * R_max_ratio )
    coil_h  = d_co + H_ratio * m_h
    t0      = T_ratio * coil_h
    
    N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))

    if two_coils:
        P = calc_power_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    else:
        P = calc_power(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
        
    if printing:
        print "HR_ratio = %.2f, R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, coil_r1 = %.2f, coil_r2 = %.2f, coil_h = %.2f, m_r = %.2f, m_h = %.2f, N = %d, P = %.2f mW" % (HR_ratio, R_ratio, H_ratio, T_ratio, coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, N, P*1000)
    return P,

toolbox.register("evaluate", evalOneMax, m_V = m_V, HR_max_ratio = HR_max_ratio, R_max_ratio = R_max_ratio, gap = gap, k_co = k_co, d_co = d_co , a = 10.0, f = 50.0, m_Br = m_Br, two_coils = two_coils, printing = False)
#toolbox.register("evaluate", evalOneMax, m_V = m_V, HR_max_ratio = 2.0, R_max_ratio = 3.0, gap = 1.26e-3, k_co = 0.6, d_co = 100e-6 , a = 10.0, f = 50.0, m_Br = 1.3, two_coils = True, printing = False)
toolbox.register("mate", tools.cxUniform, indpb=0.8)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.25)
#toolbox.register("select", tools.selRoulette)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max)) 

POPULATION = 100

population = toolbox.population(n=POPULATION)

NGEN=20
for gen in range(NGEN):

    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    top100 = tools.selBest(population, k=POPULATION)
    top = np.array(top100)
#    print "\nround finished!"
    P_max, = evalOneMax(individual = top100[0], m_V = m_V, HR_max_ratio = HR_max_ratio, R_max_ratio = R_max_ratio, gap = gap, k_co = k_co, d_co = d_co , a = 10.0, f = 50.0, m_Br = m_Br, two_coils = two_coils, printing = True)
    
#fig, axes = plt.subplots(facecolor='white', figsize=(17, 9))


HR_ratio = top[0, 0]
R_ratio  = top[0, 1]
H_ratio  = top[0, 2]
T_ratio  = top[0, 3]

m_r = ( m_V / ( np.pi * HR_max_ratio * HR_ratio ) ) ** (1./3.)
m_h =   m_V / ( np.pi * m_r * m_r )

coil_r1 = m_r + gap
coil_r2 = d_co + coil_r1 * ( 1 + R_ratio * R_max_ratio )
coil_h  = d_co + H_ratio * m_h
t0      = T_ratio * coil_h

N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))

draw_flux_lines_coil("M_fixed_volume_optimum.pdf", m_Br, m_r, m_h, coil_r1, coil_r2, coil_h, N, d_co, t0, P_max*1000, two_coils)

#top10 = tools.selBest(population, k=10)
#print "\nBest individual after optimization:"
#print "x = %.2f, y = %.2f" % (top10[0][0]*xmax, top10[0][1]*ymax)
