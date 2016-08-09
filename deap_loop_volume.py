from __future__ import division
import random
from deap import creator, base, tools, algorithms

import numpy as np
import matplotlib.pyplot as plt
import time

from magnet_flux import *

ratio_min = 0.03
#ratio_max = 1.0 - ratio_min
ratio_max = 0.99

m_V = 0.01**3 # 1 cm^3

HR_max_ratio = 3.0
gap = 0.5e-3
k_co = 0.6
d_co = 40e-6
m_Br = 1.1
PARAMS = 4

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("coordinate", random.uniform, ratio_min, ratio_max)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.coordinate, n=PARAMS)
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

def evalOneMax(individual, volume, HR_max_ratio, gap, k_co, d_co, a, f, m_Br, two_coils, printing):
    HR_ratio = individual[0]
    R_ratio  = individual[1]
    H_ratio  = individual[2]
    T_ratio  = individual[3]
    
    coil_r2 = (volume / (np.pi * HR_max_ratio * HR_ratio) ) ** (1./3.)
    coil_r1 = R_ratio * coil_r2
    if coil_r2-coil_r1 < d_co:
        coil_r1 = coil_r2 - d_co
    m_r     = coil_r1 - gap
    if m_r <= 0:
        m_r = coil_r1 * 0.01;
    
    h      = volume / (np.pi * coil_r2 * coil_r2)
    coil_h = H_ratio * h
    
    if coil_h < d_co:
        coil_h = d_co
        
    t0     = T_ratio * coil_h
    
    if two_coils:
        m_h    = h - 2 * coil_h + 2 * t0
    else:
        m_h    = h - coil_h + t0
    
    if m_h < d_co:
        m_h = d_co
    
    N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))

    if two_coils:
        P = calc_power_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    else:
        P = calc_power(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
        
    if printing:
        print "HR_ratio = %.2f, R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, coil_r1 = %.2f, coil_r2 = %.2f, coil_h = %.2f, m_r = %.2f, m_h = %.2f, m_r/m_h = %.2f, h = %.2f, N = %d, P = %.2f mW" % (HR_ratio, R_ratio, H_ratio, T_ratio, coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, m_r/m_h, h*1000, N, P*1000)
    return P,

toolbox.register("mate", tools.cxUniform, indpb=0.8)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.25)
#toolbox.register("select", tools.selRoulette)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max)) 

POPULATION = 200
NGEN=30

MAXVOL = 20
P1 = np.zeros(MAXVOL)
P2 = np.zeros(MAXVOL)
VV = np.zeros(MAXVOL)

for i in range(MAXVOL):
    m_V = (i+1) * 0.1* 0.01 ** 3
    
    two_coils = False
    toolbox.register("evaluate", evalOneMax, volume = m_V, HR_max_ratio = HR_max_ratio, gap = gap, k_co = k_co, d_co = d_co , a = 10.0, f = 50.0, m_Br = m_Br, two_coils = two_coils, printing = False)    
    population = toolbox.population(n=POPULATION)    
    
    P = 0.0
    for gen in range(NGEN):

        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))


        top100 = tools.selBest(population, k=POPULATION)
        top = np.array(top100)

        P = evalOneMax(individual = top100[0], volume = m_V, HR_max_ratio = HR_max_ratio, gap = gap, k_co = k_co, d_co = d_co , a = 10.0, f = 50.0, m_Br = m_Br, two_coils = two_coils, printing = True)[0]

    P1[i] = 1000*P
    VV[i] = m_V/(0.01**3)
    print "One coil:  m_V = %.2f cm3, P_max = %.2f mW" % (m_V/(0.01**3), P*1000)
    
    two_coils = True
    toolbox.register("evaluate", evalOneMax, volume = m_V, HR_max_ratio = HR_max_ratio, gap = gap, k_co = k_co, d_co = d_co , a = 10.0, f = 50.0, m_Br = m_Br, two_coils = two_coils, printing = False)    
    population = toolbox.population(n=POPULATION)    
    
    P = 0.0
    for gen in range(NGEN):

        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))


        top100 = tools.selBest(population, k=POPULATION)
        top = np.array(top100)

        P = evalOneMax(individual = top100[0], volume = m_V, HR_max_ratio = HR_max_ratio, gap = gap, k_co = k_co, d_co = d_co , a = 10.0, f = 50.0, m_Br = m_Br, two_coils = two_coils, printing = True)[0]

    P2[i] = 1000*P
    print "Two coils:  m_V = %.2f cm3, P_max = %.2f mW" % (m_V/(0.01**3), P*1000)


fig, axes = plt.subplots(facecolor='white', figsize=(17, 9))
plt.plot(VV, P1, label="One Coil")
plt.plot(VV, P2, label="Two Coils")
plt.legend(loc=2)
plt.show(block=False)
raw_input("daa")

fig, axes = plt.subplots(facecolor='white', figsize=(17, 9))
plt.plot(VV, np.divide(P2,P1), label=r"$\frac{\textit{two coils}}{\textit{one coil}}$")
plt.legend(loc=2)
plt.show(block=False)

raw_input("daa")

