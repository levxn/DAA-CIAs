import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load the dataset
data = pd.read_csv('Levin/SNU/Sem4/DAA/Bank_Personal_Loan_Modelling.csv')

# Split the data into features and target
X = data.drop('Personal Loan', axis=1).values
y = data['Personal Loan'].values

# Define the objective function


def objective_function(position):
    X_subset = X[:, position == 1]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_subset, y)
    score = clf.score(X_subset, y)
    return score

# Define the ACO algorithm


class ACO:
    def __init__(self, num_ants, num_dimensions, max_iterations, alpha=1.0, beta=5.0, rho=0.5, q0=0.5):
        self.num_ants = num_ants
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.pheromone = np.ones((num_ants, num_dimensions)) / num_dimensions
        self.best_solution = None
        self.best_score = -1

    def run(self, verbose=True):
        for i in range(self.max_iterations):
            solutions = np.zeros((self.num_ants, self.num_dimensions))
            scores = np.zeros(self.num_ants)

            # Generate ant solutions
            for ant in range(self.num_ants):
                positions = np.zeros(self.num_dimensions)
                for j in range(self.num_dimensions):
                    positions = self.select_next_feature(ant, positions)
                solutions[ant] = positions
                scores[ant] = self.objective_function(positions)

                # Update best solution
                if scores[ant] > self.best_score:
                    self.best_score = scores[ant]
                    self.best_solution = positions

            # Update pheromone trails
            delta_pheromone = np.zeros((self.num_ants, self.num_dimensions))
            for ant in range(self.num_ants):
                for j in range(self.num_dimensions):
                    if solutions[ant][j] == 1:
                        delta_pheromone[ant][j] = 1 / scores[ant]
            self.pheromone = (1 - self.rho) * self.pheromone + \
                self.rho * delta_pheromone

            if verbose:
                print(f'Iteration {i+1}: Best score = {self.best_score}')

    def select_next_feature(self, ant, positions):
        remaining_features = np.where(positions == 0)[0]

        # Calculate heuristic information
        heuristic = np.zeros(self.num_dimensions)
        heuristic[remaining_features] = 1 / \
            (1 + np.exp(-self.beta * self.objective_function(positions + self.alpha)))

        # Calculate probability distribution
        probability = self.pheromone[ant][remaining_features] ** self.alpha * \
            heuristic[remaining_features] ** self.beta
        probability = probability / np.sum(probability)

        # Choose next feature based on probability
        if np.random.uniform() < self.q0:
            next_feature = remaining_features[np.argmax(probability)]
        else:
            next_feature = np.random.choice(remaining_features, p=probability)

        positions[next_feature] = 1
        return positions


# Set ACO parameters
num_ants = 10
num_dimensions = X.shape[1]
max_iterations = 50
alpha = 1.
beta = 2.
rho = 0.5

# Define objective function


def objective_function(positions):
    X_selected = X[:, positions.astype(bool)].ravel()
    clf = DecisionTreeClassifier(random_state=0)
    scores = cross_val_score(clf, X_selected, y, cv=10)
    return np.mean(scores)

# Define ACO class


class AntColony:
    def __init__(self, num_ants, num_dimensions, max_iterations, alpha, beta, rho, objective_function):
        self.num_ants = num_ants
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.objective_function = objective_function

        # Initialize pheromone levels to 1
        self.pheromone = np.ones((num_ants, num_dimensions))

    def select_next_feature(self, ant, positions):
        remaining_features = np.where(positions[ant] == 0)[0]

        # Calculate heuristic information
        heuristic = np.zeros(self.num_dimensions)
        heuristic[remaining_features] = 1 / \
            (1 + np.exp(-self.beta *
             self.objective_function(positions[ant] + self.alpha)))

        # Calculate probability distribution
        probability = self.pheromone[ant][remaining_features] ** self.alpha * \
            heuristic[remaining_features] ** self.beta
        probability = probability / np.sum(probability)

        # Choose next feature based on probability
        next_feature = np.random.choice(remaining_features, p=probability)
        positions[ant, next_feature] = 1

        return positions

    def run(self):
        best_solution = None
        best_score = -np.inf

        for i in range(self.max_iterations):
            # Initialize ant positions
            ant_positions = np.zeros((self.num_ants, self.num_dimensions))

            # Generate solutions
            for ant in range(self.num_ants):
                for j in range(self.num_dimensions):
                    ant_positions = self.select_next_feature(
                        ant, ant_positions)

                score = self.objective_function(ant_positions)

                # Update best solution
                if score > best_score:
                    best_solution = ant_positions
                    best_score = score

                # Update pheromone levels
                self.pheromone[ant][ant_positions[ant].astype(
                    bool)] += 1 / (1 + score)

            # Evaporate pheromone
            self.pheromone *= (1 - self.rho)

        return best_solution, best_score


# Run ACO algorithm
aco = AntColony(num_ants=num_ants,
                num_dimensions=num_dimensions,
                max_iterations=max_iterations,
                alpha=alpha,
                beta=beta,
                rho=rho,
                objective_function=objective_function)
best_solution, best_score = aco.run()

# Print selected features
selected_features = np.where(best_solution[0] == 1)[0]
selected_columns = X.columns[selected_features]
print("Selected features: ", selected_columns)
