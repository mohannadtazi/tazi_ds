# `LinearRegression` Class Documentation
## Overview
The LinearRegression class in the `tazids` library implements a simple linear regression model using gradient descent. This class allows you to fit a model to data, compute predictions, and monitor the learning process via the loss function. The model learns the optimal parameters (slope a and intercept b) to best fit the input data using the Mean Squared Error (MSE) cost function and gradient descent optimization.

## Installation
To install the tazids library, you can use the following command:

```bash
pip install tazids
```
Class: `LinearRegression`
Constructor: `__init__(self)`
The constructor initializes the model's parameters. It sets up an empty dictionary, parameters, which will hold the values for the model's parameters (slope a and intercept b).

```python
def __init__(self):
    self.parameters = {}
```
Method: `fw_prop(self, X)`
This method performs forward propagation, predicting the target variable y_pred using the model's parameters (a and b) and the given input features X. The linear regression equation is used:

![image](https://github.com/user-attachments/assets/87b4bb7d-cfc6-4f94-bf3f-cf7d40a7af5b)

**Parameters**:
X: The input feature data (numpy array or similar).
**Returns**:
y_pred: The predicted target values.
```python
def fw_prop(self, X):
    a = self.parameters['a']
    b = self.parameters['b']
    y_pred = a * X + b
    return y_pred
```
Method: `cost_function(self, y, y_pred)`
This method calculates the cost using the Mean Squared Error (MSE) formula:

![image](https://github.com/user-attachments/assets/da67e0fb-63b8-46fc-a36b-99ca80c63739)

**Parameters**:
y: The actual target values.
y_pred: The predicted target values.
**Returns**:
cost: The mean squared error between y and y_pred.
```python
def cost_function(self, y, y_pred):
    cost = np.mean((y_pred - y) ** 2)
    return cost
```
Method: `back_prop(self, X, y, y_pred)`
This method computes the gradients (derivatives) of the cost function with respect to the parameters (a and b). These gradients are needed to update the parameters during the optimization process.

**Parameters**:
X: The input features.
y: The actual target values.
y_pred: The predicted target values.
**Returns**:
derivatives: A dictionary containing the gradients for the parameters (da for a, db for b).
```python
def back_prop(self, X, y, y_pred):
    derivatives = {}
    df = y_pred - y
    derivatives['da'] = 2 * np.mean(X * df)
    derivatives['db'] = 2 * np.mean(df)
    return derivatives
```
Method: `update_params(self, derivatives, learning_rate)`
This method updates the model parameters (a and b) using the gradients computed in back_prop. The update is done using the gradient descent rule:

![image](https://github.com/user-attachments/assets/86e26606-6c52-4616-9d1a-8c66c0b1684b)

**Parameters**:
derivatives: A dictionary containing the gradients for the parameters (da and db).
learning_rate: The step size for updating the parameters.
```python
def update_params(self, derivatives, learning_rate):
    self.parameters['a'] -= learning_rate * derivatives['da']
    self.parameters['b'] -= learning_rate * derivatives['db']
```
Method: `fit(self, X, y, learning_rate=0.1, iters=1000)`
This method trains the linear regression model using gradient descent. It initializes the parameters (a and b), performs the forward pass, calculates the cost, computes the gradients, updates the parameters, and repeats the process for the specified number of iterations.

**Parameters**:
X: The input features.
y: The target values.
learning_rate: The learning rate for gradient descent (default is 0.1).
iters: The number of iterations for training the model (default is 1000).
**Returns**:
None. The model is trained in place, and the loss is printed during training.
```python
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
```
Method: `predict(self, X)`
This method makes predictions using the trained parameters (a and b). It applies the linear regression equation to the input features X.

**Parameters**:
X: The input features.
**Returns**:
y_pred: The predicted target values.
```python
def predict(self, X):
    a = self.parameters['a']
    b = self.parameters['b']
    y_pred = a * X + b
    print("Tazi guessed the following values")
    return y_pred
```

## Example Usage
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
### Notes
- The model uses random initialization for the parameters a and b, so results may vary on each run.
- The learning rate and number of iterations can be adjusted to fine-tune the training process.
- The loss (Mean Squared Error) is printed every 100 iterations to monitor the progress of training.
- The method predict provides the model's predictions for the input data.
