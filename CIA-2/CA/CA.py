from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class CulturalAlgorithm:
    def __init__(self, num_ants, num_dimensions, max_iterations, pc, pm, objective_function, X_train, y_train):
        self.num_ants = num_ants
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.pc = pc
        self.pm = pm
        self.objective_function = objective_function
        self.X_train = X_train
        self.y_train = y_train

    def run(self):
        # Initialize population
        population = np.random.randint(
            2, size=(self.num_ants, self.num_dimensions))

        # Initialize best solution and fitness
        best_solution = None
        best_fitness = -np.inf

        # Iterate for max_iterations
        for iteration in range(self.max_iterations):
            # Evaluate fitness of population
            fitness = np.array([self.objective_function(
                positions, self.X_train, self.y_train) for positions in population])

            # Update best solution
            if np.max(fitness) > best_fitness:
                best_solution = population[np.argmax(fitness)]
                best_fitness = np.max(fitness)

            # Sort population by fitness
            sorted_indices = np.argsort(fitness)[::-1]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Perform crossover and mutation
            for i in range(self.num_ants):
                # Perform crossover
                if np.random.rand() < self.pc:
                    j = np.random.randint(self.num_ants)
                    k = np.random.randint(self.num_dimensions)
                    child = np.concatenate(
                        (population[i, :k], population[j, k:]))
                    population = np.vstack((population, child))

                # Perform mutation
                if np.random.rand() < self.pm:
                    j = np.random.randint(self.num_dimensions)
                    population[i, j] = 1 - population[i, j]

            # Remove worst solutions
            population = population[:self.num_ants, :]

        return best_solution, best_fitness


# Load data from CSV file
data = pd.read_csv("Levin/SNU/Sem4/DAA/Bank_Personal_Loan_Modelling.csv")

# Split data into features and target
X = data.drop(columns=['Personal Loan']).values
y = data['Personal Loan'].values

# Define bounds for each feature
bounds = np.zeros((X.shape[1], 2))
bounds[:, 1] = 1

# Define objective function


def objective_function(positions, X_train, y_train):
    selected_features = X_train[:, positions.astype(bool)]
    model = LogisticRegression()
    model.fit(selected_features, y_train)
    y_pred = model.predict(selected_features)
    return accuracy_score(y_train, y_pred)


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize CA algorithm
ca = CulturalAlgorithm(num_ants=100, num_dimensions=X.shape[1], max_iterations=10, pc=0.7,
                       pm=0.001, objective_function=objective_function, X_train=X_train, y_train=y_train)

# Run CA algorithm
best_solution, best_fitness = ca.run()

# Print best solution and fitness
print("Best Solution: ", best_solution)
print("Best Fitness: ", best_fitness)
