import numpy as np

class KMeans:
    def __init__(self, n_iter=300, tol=1e-4):
        self.max_iters = n_iter
        self.tolerance = tol
        self.centroids = None
        self.labels = None

    def fit(self, X, k):
        self.n_clusters = k
        print("Tazi is proud of You")
        random_indices = np.random.choice(X.shape[0], k, replace=False )
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break
            self.centroids = new_centroids

    def predict(self, X):
        print("Tazi guess ...")
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)