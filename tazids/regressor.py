import numpy as np
import random

class LinearRegression:
    def __init__(self):
        self.parameters = {}

    def fw_prop(self, X):
        a = self.parameters['a']
        b = self.parameters['b']
        y_pred = a * X + b
        return y_pred

    def cost_function(self, y, y_pred):
        cost = np.mean((y_pred - y) ** 2)
        return cost

    def back_prop(self, X, y, y_pred):
        derivatives = {}
        df = y_pred - y
        derivatives['da'] = 2 * np.mean(X * df)
        derivatives['db'] = 2 * np.mean(df)
        return derivatives

    def update_params(self, derivatives, learning_rate):
        self.parameters['a'] -= learning_rate * derivatives['da']
        self.parameters['b'] -= learning_rate * derivatives['db']

    def fit(self, X, y, learning_rate=0.1, iters=1000):
        self.parameters['a'] = np.random.uniform(-1, 1)
        self.parameters['b'] = np.random.uniform(-1, 1)
        self.loss = []
        print("Tazi is proud of you!")
        for i in range(iters):
            predictions = self.fw_prop(X)
            cost = self.cost_function(y, predictions)
            derivatives = self.back_prop(X, y, predictions)
            self.update_params(derivatives, learning_rate)
            self.loss.append(cost)
            if i % 100 == 0:
                print(f"Iteration = {i}, Loss = {cost}")

    def predict(self, X):
        a = self.parameters['a']
        b = self.parameters['b']
        y_pred = a * X + b
        print("Tazi guessed the following values")
        return y_pred
