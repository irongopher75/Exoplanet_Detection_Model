"""
Genetic Algorithm optimizer for threshold tuning in exoplanet detection.
Finds optimal decision threshold to maintain precision as SNR drops.
"""
import numpy as np

class GeneticAlgorithmOptimizer:
    def __init__(self, population_size=20, generations=50, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    def optimize(self, eval_func):
        population = np.random.uniform(0, 1, self.population_size)
        for gen in range(self.generations):
            fitness = np.array([eval_func(thresh) for thresh in population])
            idx = np.argsort(fitness)[-self.population_size//2:]
            parents = population[idx]
            children = parents + np.random.normal(0, self.mutation_rate, len(parents))
            children = np.clip(children, 0, 1)
            population = np.concatenate([parents, children])
        best_idx = np.argmax([eval_func(thresh) for thresh in population])
        return population[best_idx]
