import numpy as np
import pandas as pd


class GeneticAlgorithm:
    def __init__(self, population_size, num_generations, crossover_rate, mutation_rate, elitism_rate):
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate

    def run(self, filename):
        # Load data
        df = pd.read_csv(filename)

        # Extract features and target variable
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Initialize population
        population = np.random.randint(
            2, size=(self.population_size, X.shape[1]))

        # Initialize best solution and fitness
        best_solution = None
        best_fitness = -np.inf

        # Iterate for num_generations
        for generation in range(self.num_generations):
            # Evaluate fitness of population
            fitness = np.array([self.fitness(solution, X, y)
                               for solution in population])

            # Update best solution and fitness
            if np.max(fitness) > best_fitness:
                best_solution = population[np.argmax(fitness)]
                best_fitness = np.max(fitness)

            # Perform elitism
            elites_count = int(self.elitism_rate * self.population_size)
            elites_indices = np.argsort(fitness)[::-1][:elites_count]
            elites = population[elites_indices]

            # Create the mating pool
            non_elites_count = self.population_size - elites_count
            mating_pool_indices = np.random.choice(range(
                self.population_size), size=non_elites_count, replace=True, p=fitness/np.sum(fitness))
            mating_pool = population[mating_pool_indices]

            # Create the next generation population by combining the elites and the offspring from the mating pool
            offspring = self.crossover(mating_pool)
            offspring = self.mutate(offspring)
            population = np.vstack((elites, offspring))

        return best_solution, best_fitness

    def fitness(self, solution, X, y):
        y_pred = np.sum(solution * X, axis=1)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def crossover(self, mating_pool):
        offspring = np.zeros_like(mating_pool)

        for i in range(mating_pool.shape[0]):
            # Choose two parents randomly
            parent_indices = np.random.choice(
                mating_pool.shape[0], size=2, replace=False)
            parent1 = mating_pool[parent_indices[0]]
            parent2 = mating_pool[parent_indices[1]]

            # Perform crossover
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, mating_pool.shape[1])
                offspring[i, :crossover_point] = parent1[:crossover_point]
                offspring[i, crossover_point:] = parent2[crossover_point:]
            else:
                offspring[i] = parent1

        return offspring

    def mutate(self, offspring):
        for i in range(offspring.shape[0]):
            # Perform mutation
            for j in range(offspring.shape[1]):
                if np.random.rand() < self.mutation_rate:
                    offspring[i, j] = 1 - offspring[i, j]

        return offspring


if __name__ == '__main__':
    ga = GeneticAlgorithm(population_size=100, num_generations=50,
                          crossover_rate=0.8, mutation_rate=0.01, elitism_rate=0.1)
    best_solution, best_fitness = ga.run(
        filename='Levin/SNU/Sem4/DAA/Bank_Personal_Loan_Modelling.csv')
    print('Best solution:', best_solution)
    print('Best fitness:', best_fitness)
