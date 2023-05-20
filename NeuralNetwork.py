import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.neurons = []
        self.weights = []
        self.fitness = 0.0

        self.init_neurons()
        self.init_weights()

    def get_weights(self):
        return self.weights

    def init_neurons(self):
        self.neurons = [np.zeros(layer) for layer in self.layers]

    def init_weights(self):
        self.weights = []
        for i in range(1, len(self.layers)):
            layer_weights = [np.random.uniform(-0.5, 0.5, self.layers[i - 1]) for _ in range(self.layers[i])]
            self.weights.append(layer_weights)

    def copy_weights(self, copy_weights):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = copy_weights[i][j][k]

    def feed_forward(self, inputs):
        self.neurons[0] = np.array(inputs)
        for i in range(1, len(self.layers)):
            for j in range(len(self.neurons[i])):
                value = np.dot(self.weights[i - 1][j], self.neurons[i - 1])
                self.neurons[i][j] = np.tanh(value)

        return self.neurons[-1]

    def mutate(self, modifier):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    weight = self.weights[i][j][k]
                    random_number = np.random.uniform(0, 1000*modifier)

                    if random_number <= 2:
                        weight *= -1
                    elif random_number <= 4:
                        weight = np.random.uniform(-0.5, 0.5)
                    elif random_number <= 6:
                        factor = np.random.uniform(0, 1) + 1
                        weight *= factor
                    elif random_number <= 8:
                        factor = np.random.uniform(0, 1)
                        weight *= factor

                    self.weights[i][j][k] = weight

    def add_fitness(self, fit):
        self.fitness += fit

    def set_fitness(self, fit):
        self.fitness = fit

    def get_fitness(self):
        return self.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness