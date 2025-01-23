# Tazids Library Documentation
# Tazids Library

![PyPI](https://img.shields.io/pypi/v/tazids)
![Python](https://img.shields.io/pypi/pyversions/tazids)
![Downloads](https://img.shields.io/pypi/dm/tazids)

![image](https://github.com/user-attachments/assets/cfc285b8-8d66-4bd9-9265-f875eb8ea9bf)

## Overview
Tazids is a simple yet powerful machine learning library that provides implementations for various algorithms, including Linear Regression, Decision Trees, and K-Means clustering. The library is designed to help you easily build, train, and apply machine learning models for a variety of tasks, with a focus on transparency and simplicity.

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
│   ├── clustering.py      # Contains the KMeans class
└── setup.py               # Setup script for installation
```

## Features  
### `Linear Regression` Class (LinearRegression)
The **LinearRegression** class implements a simple linear regression model using gradient descent. It allows you to:  
- Train the model on data using gradient descent.  
- Predict outcomes for new data.  
- Monitor the training process with loss values.  

#### Example Usage
```python
import numpy as np
from tazids.regressor import LinearRegression

# Generate synthetic data
X = np.random.rand(100, 1) * 10
y = 5 * X + np.random.randn(100, 1) * 2  # Linear relationship with noise

# Initialize and train the model
model = LinearRegression()
model.fit(X, y, learning_rate=0.01, iters=1000)

# Make predictions
predictions = model.predict(X)
```

### `Decision Tree` Class (DecisionTree)
The **DecisionTree** class implements a simple decision tree for classification tasks. It supports:  
- Recursive splitting of the data to build the decision tree.  
- Making predictions by traversing the tree for each input sample.  

#### Example Usage  

```python
from tazids.tree import DecisionTree
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Initialize and train the model
model = DecisionTree()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

### `KMeans` Class (KMeans)
The KMeans class implements the K-Means clustering algorithm for unsupervised learning. It clusters data points into k groups by:  

- Initializing random centroids.  
- Assigning points to the nearest centroid.  
- Updating centroids iteratively until convergence.  
### Example Usage
```python
from tazids.clustering import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Initialize and train the KMeans model
model = KMeans(n_iter=300, tol=1e-4)
model.fit(X, k=4)

# Predict cluster labels
predictions = model.predict(X)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis', s=30)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.legend()
plt.title("KMeans Clustering")
plt.show()
```

## Notes
- Tazids is designed to be simple and transparent for educational and research purposes.
- Transparency: All models are built from scratch to provide a better understanding of their inner workings.
- Learning-Oriented: Ideal for students and researchers who want to explore the basics of machine learning.
- Lightweight: Minimal dependencies to keep the library easy to use.
- For advanced use cases, consider established libraries like scikit-learn, but Tazids offers a perfect entry point to learn the fundamentals.

# Made By [Mohannad Tazi](https://www.linkedin.com/in/mohannad-tazi/) 
