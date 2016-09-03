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

V = 0.01**3 # 1 cm^3

m_r = 9.525e-3/2
m_h = 19.05e-3
m_V = np.pi * m_r * m_r * m_h
print "V = %.e, m_V = %.e, m_V/V = %.2f" % (V, m_V, m_V/V)

m_Br = 1.3
gap = 1.26e-3
R_max_ratio = 2.0
two_coils = True
k_co = 0.6
d_co = 100e-6


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

def evalOneMax(individual, m_r, m_h, R_max_ratio, gap, k_co, d_co, a, f, m_Br, two_coils, printing):
    R_ratio  = individual[0]
    H_ratio  = individual[1]
    T_ratio  = individual[2]
    
    coil_r1 = m_r + gap
    coil_r2 = coil_r1 * ( 1 + R_ratio * R_max_ratio )
    if coil_r2 - coil_r1 < d_co:
        coil_r2 = coil_r1 + d_co
    coil_h  = H_ratio * m_h
    t0      = T_ratio * coil_h
    
    N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))

    if two_coils:
        P = calc_power_two_coils(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
    else:
        P = calc_power(m_Br, m_h, m_r, coil_h, coil_r1, coil_r2, N, d_co, t0, a, f)
        
    if printing:
        print "R_ratio = %.2f, H_ratio = %.2f, T_ratio = %.2f, coil_r1 = %.2f, coil_r2 = %.2f, coil_h = %.2f, m_r = %.2f, m_h = %.2f, m_r/m_h = %.2f, N = %d, P = %.2f mW" % (R_ratio, H_ratio, T_ratio, coil_r1*1000, coil_r2*1000, coil_h*1000, m_r*1000, m_h*1000, m_r/m_h, N, P*1000)
    return P,

toolbox.register("evaluate", evalOneMax, m_r = m_r, m_h = m_h, R_max_ratio = R_max_ratio, gap = gap, k_co = k_co, d_co = d_co , a = 10.0, f = 50.0, m_Br = m_Br, two_coils = two_coils, printing = False)
toolbox.register("mate", tools.cxUniform, indpb=0.8)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.25)
#toolbox.register("select", tools.selRoulette)
PARAMS = 3
toolbox.register("select", tools.selTournament, tournsize=PARAMS)

toolbox.decorate("mate", checkBounds(ratio_min, ratio_max))
toolbox.decorate("mutate", checkBounds(ratio_min, ratio_max)) 

POPULATION = 100

population = toolbox.population(n=POPULATION)

NGEN=20

gen_max = np.ones(NGEN)

evolution = np.zeros((PARAMS+1, POPULATION, NGEN))

for gen in range(NGEN):
    
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    pop = np.array(population)

    evolution[0, :, gen] = pop[:,0]
    evolution[1, :, gen] = pop[:,1]
    evolution[2, :, gen] = pop[:,2]
    
    np_fits = np.array(fits)*1000
    evolution[3, :, gen] = np_fits.flatten()    
    
    top100 = tools.selBest(population, k=POPULATION)
    top = np.array(top100)
#    print "\nround finished!"
    gen_max[gen] = 1000*evalOneMax(individual = top100[0], m_r = m_r, m_h = m_h, R_max_ratio = R_max_ratio, gap = gap, k_co = k_co, d_co = d_co , a = 10.0, f = 50.0, m_Br = m_Br, two_coils = two_coils, printing = True)[0]


fig, axes = plt.subplots(facecolor='white', nrows=4, ncols=1, figsize=(17, 9))

axes[0].boxplot(evolution[0, :, :], whis='range')
axes[0].set_title("R_ratio")

axes[1].boxplot(evolution[1, :, :], whis='range')
axes[1].set_title("H_ratio")

axes[2].boxplot(evolution[2, :, :], whis='range')
axes[2].set_title("T_ratio")

axes[3].boxplot(evolution[3, :, :], whis='range')
axes[3].set_title(r"$P_\mathrm{max}\,\,(\mathrm{mW})$")

fig.subplots_adjust(wspace=0.01, hspace=0.4, left=0.05, right=0.98, top=0.97, bottom=0.02)
plt.show(block=False)
raw_input("hit any key!")
plt.savefig("optim_progress.pdf")
plt.close()

#fig, axes = plt.subplots(facecolor='white', figsize=(17, 9))
R_ratio  = top[0, 0]
H_ratio  = top[0, 1]
T_ratio  = top[0, 2]

coil_r1 = m_r + gap
coil_r2 = coil_r1 * ( 1 + R_ratio * R_max_ratio )
coil_h  = H_ratio * m_h
t0      = T_ratio * coil_h
N = int(round( 4.0 * coil_h * (coil_r2 - coil_r1) * k_co / (d_co*d_co*np.pi) ))

draw_flux_lines_coil("M_fixed_optimum.pdf", m_Br, m_r, m_h, coil_r1, coil_r2, coil_h, N, d_co, t0, gen_max[NGEN-1], two_coils)



#top10 = tools.selBest(population, k=10)
#print "\nBest individual after optimization:"
#print "x = %.2f, y = %.2f" % (top10[0][0]*xmax, top10[0][1]*ymax)
