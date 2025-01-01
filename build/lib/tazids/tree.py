import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.root=None


    def _gini_impurity(self, labels):
        unique, count = np.unique(labels, return_counts=True)
        probabilities = count / count.sum()
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _impurity_split(self,parent, d_left, d_right):
        left_impurity = self._gini_impurity(d_left)
        left_weight = len(d_left) / len(parent)
        right_impurity = self._gini_impurity(d_right)
        right_weight = len(d_right) / len(parent)
        total_impurity = (left_weight * left_impurity) + (right_weight * right_impurity)
        print(f"Parent: {parent}, Left: {d_left}, Right: {d_right}, Impurity: {total_impurity}")
    
        return total_impurity

    def _information_gain(self, parent, d_left, d_right):
        parent_impurity= self._gini_impurity(parent)
        split_impurity = self._impurity_split(parent, d_left, d_right)
        info_gain = parent_impurity - split_impurity
    
        return info_gain

    def _split(self, dataset, feature_index, threshold):
        left = dataset[dataset[:, feature_index] <= threshold]
        right = dataset[dataset[:, feature_index] > threshold]
        return left, right

    def _best_split(self, dataset):
        X = dataset[:, :-1]
        y = dataset[:, -1]
        
        num_samples, num_features = X.shape

        max_IG = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            
            if len(unique_values) > 1:  # Check if feature is continuous
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoint thresholds
            else:
                thresholds = unique_values  # If categorical, just use the unique values as thresholds

            for threshold in thresholds:
                left, right = self._split(dataset, feature, threshold)
                if len(left) == 0 or len(right) == 0:
                    continue

                IG = self._information_gain(dataset, left[:, -1], right[:, -1])
                if IG > max_IG:
                    max_IG = IG
                    best_feature = feature
                    best_threshold = threshold
                    
        print(f"Best Feature: {best_feature}, Best Threshold: {best_threshold}")
        return best_feature, best_threshold


    def _majority_class(self, labels):
        return np.argmax(np.bincount(labels.astype(int)))

    def _build_tree(self,dataset, depth=0):
        X = dataset[:, :-1]
        y = dataset[:, -1]
        print(f"Building tree at depth {depth}, Samples: {len(y)}")


        if len(np.unique(y)) == 1 or len(y) == 0 or (self.max_depth and depth >= self.max_depth) or len(y) < self.min_samples_split:
            print(f"Stopping at depth {depth}. Majority class: {self._majority_class(y)}")
            return Node(value=self._majority_class(y))
            
        best_feature, best_threshold = self._best_split(dataset)
        
        if best_feature is None:
            print(f"Enouugh😫! No split possible at depth {depth}. Majority class: {self._majority_class(y)}")
            return Node(value=self._majority_class(y))
            
        print(f"✂️ Splitting on feature {best_feature} at threshold {best_threshold} at depth {depth}")
        left, right = self._split(dataset, best_feature, best_threshold)
        left_child = self._build_tree(left, depth + 1)
        right_child = self._build_tree(right, depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            print(f"🍀 Leaf node reached. Returning value: {node.value}")
            return node.value

        if x[node.feature] <= node.threshold:
            print(f"👈🏽 Traversing left at node {node.feature} with threshold {node.threshold}")
            return self._traverse_tree(x, node.left)
        else:
            print(f"👉🏽Traversing right at node {node.feature} with threshold {node.threshold}")
            return self._traverse_tree(x, node.right)
            
        
    def fit(self, X, y):
        dataset = np.c_[X, y]
        print("Tazi is proud of you!")
        self.root = self._build_tree(dataset)

    def predict(self, X):
        print("Tazi guessed the following labels")
        return np.array([self._traverse_tree(x, self.root) for x in X])
        
        
        