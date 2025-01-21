# Tazids Library Documentation
# Tazids Library

![PyPI](https://img.shields.io/pypi/v/tazids)
![Python](https://img.shields.io/pypi/pyversions/tazids)
![Downloads](https://img.shields.io/pypi/dm/tazids)

## Overview
Tazids is a simple yet powerful machine learning library that provides implementations for various algorithms, including Linear Regression and Decision Trees. The library is designed to help you easily build and train models for regression and classification tasks, with a focus on transparency and simplicity.

## Installation
To install the Tazids library, you can use the following command:
```bash
pip install tazids
```

## Directory Structure
The Tazids library is organized as follows:
```
Tazi_Ds/
├── tazids/
│   ├── __init__.py        # Initialization file for the tazids module
│   ├── regressor.py       # Contains the LinearRegression class
│   ├── tree.py            # Contains the DecisionTree class
└── setup.py               # Setup script for installation
```

## `Linear Regression` Class (LinearRegression)
The **LinearRegression** class implements a simple linear regression model using gradient descent. It allows you to fit a model to data, compute predictions, and monitor the learning process.

### Constructor: `__init__(self)`
The constructor initializes the model's parameters.
```python
class LinearRegression:
    def __init__(self):
        self.parameters = {}
```

### Method: `fw_prop(self, X)`
Performs forward propagation to predict the target variable using the model's parameters (slope `a` and intercept `b`).
```python
    def fw_prop(self, X):
        a = self.parameters['a']
        b = self.parameters['b']
        y_pred = a * X + b
        return y_pred
```

### Method: `cost_function(self, y, y_pred)`
Calculates the cost using the Mean Squared Error (MSE) formula.
```python
    def cost_function(self, y, y_pred):
        cost = np.mean((y_pred - y) ** 2)
        return cost
```

### Method: `back_prop(self, X, y, y_pred)`
Computes the gradients (derivatives) of the cost function with respect to the parameters.
```python
    def back_prop(self, X, y, y_pred):
        derivatives = {}
        df = y_pred - y
        derivatives['da'] = 2 * np.mean(X * df)
        derivatives['db'] = 2 * np.mean(df)
        return derivatives
```

### Method: `update_params(self, derivatives, learning_rate)`
Updates the model parameters using the gradients computed during the backpropagation.
```python
    def update_params(self, derivatives, learning_rate):
        self.parameters['a'] -= learning_rate * derivatives['da']
        self.parameters['b'] -= learning_rate * derivatives['db']
```

### Method: `fit(self, X, y, learning_rate=0.1, iters=1000)`
Trains the linear regression model using gradient descent.
```python
    def fit(self, X, y, learning_rate=0.1, iters=1000):
        self.parameters['a'] = np.random.uniform(-1, 1)
        self.parameters['b'] = np.random.uniform(-1, 1)
        self.loss = []
        for i in range(iters):
            predictions = self.fw_prop(X)
            cost = self.cost_function(y, predictions)
            derivatives = self.back_prop(X, y, predictions)
            self.update_params(derivatives, learning_rate)
            self.loss.append(cost)
            if i % 100 == 0:
                print(f"Iteration = {i}, Loss = {cost}")
```

### Method: `predict(self, X)`
Makes predictions using the trained parameters.
```python
    def predict(self, X):
        a = self.parameters['a']
        b = self.parameters['b']
        y_pred = a * X + b
        return y_pred
```

### Example Usage
```python
import numpy as np
from tazids.regressor import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Random features between 0 and 10
y = np.random.rand(100) * 100    # Random target variable between 0 and 100

# Initialize and train the model
model = LinearRegression()
model.fit(X, y, learning_rate=0.001, iters=1000)

# Make predictions
predictions = model.predict(X)
```

## `Decision Tree` Class (DecisionTree)
The **DecisionTree** class implements a simple decision tree algorithm for classification tasks. It builds a tree based on the input features and predicts the class of a given input.

### Constructor: `__init__(self)`
Initializes the decision tree parameters.
```python
class DecisionTree:
    def __init__(self):
        self.tree = None
```

### Method: `fit(self, X, y)`
Builds the decision tree by recursively splitting the dataset at the best feature and threshold.
```python
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
```

### Method: `_build_tree(self, X, y)`
Recursively builds the tree by finding the best feature to split the data at each node.
```python
    def _build_tree(self, X, y):
        # Recursive function to build the decision tree
        pass
```

### Method: `predict(self, X)`
Makes predictions by traversing the decision tree for each input sample.
```python
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
```

### Example Usage
```python
from tazids.tree import DecisionTree
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Initialize and train the model
model = DecisionTree()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## Notes
- Tazids is designed to be simple and transparent for educational and research purposes.
- Both the `LinearRegression` and `DecisionTree` models are built from scratch, showcasing the basic principles behind these algorithms.
- For more complex use cases, consider using established libraries like scikit-learn, but Tazids is a great way to learn how these models work under the hood.

# Made By [Mohannad Tazi](https://www.linkedin.com/in/mohannad-tazi/) 
