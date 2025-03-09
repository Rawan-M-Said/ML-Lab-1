import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth                  # Maximum depth of the tree can be reached
        self.min_samples_split = min_samples_split  # Least number of samples required to split a node
        self.tree = None
    
    def fit(self, X, y, sample_weights=None):
        # Calculating the weights if not provided
        if sample_weights is None:
            # each sample has a weight of 1/n_samples
            sample_weights = np.ones(len(y))
        # Constructing the tree
        self.tree = self._grow_tree(X, y, sample_weights, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(sample, self.tree) for sample in X])

    def _grow_tree(self, X, y, sample_weights, depth):
        n_samples, n_features = X.shape

        # Stopping criteria
        # If the depth limit is reached or the number of samples is less than the minimum required for
        #  a split or all the samples have the same label then create a leaf node
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1:
            return self._create_leaf(y, sample_weights)
        
        # Finding the best split
        best_feature, best_threshold = self._best_split(X, y, sample_weights)
        # If no split is found, create a leaf node
        if best_feature is None:
            return self._create_leaf(y, sample_weights)

        # Split the dataset
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        left = self._grow_tree(X[left_idx], y[left_idx], sample_weights[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], sample_weights[right_idx], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left, "right": right}
       
    def _create_leaf(self, y, sample_weights):
        # Create a leaf node
        # The leaf node will contain the most common label in the samples
        class_counts = {}
        total_weight = np.sum(sample_weights)
        for label in np.unique(y):
            class_counts[label] = np.sum(sample_weights[np.where(y == label)]) / total_weight

        return {"leaf": max(class_counts, key=class_counts.get)}

    def _best_split(self, X, y, sample_weights):
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Looping through all the features and their unique values to find the best split
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, sample_weights, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, X, y, sample_weights, feature, threshold):
        parent_entropy = self._entropy(y, sample_weights)
        # Splitting the dataset
        # left_idx contains the indices of the samples that are less than or equal to the threshold
        # and right_idx contains the indices of the samples that are greater than the threshold
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        # If all the samples are in one side of the split, then the split is not valid
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return 0
        
        # Calculating the entropy of the left and right nodes
        left_entropy = self._entropy(y[left_idx], sample_weights[left_idx])
        right_entropy = self._entropy(y[right_idx], sample_weights[right_idx])

        # Calculating the weight of the left and right nodes as a fraction of the total weight
        left_weight = np.sum(sample_weights[left_idx]) / np.sum(sample_weights)
        right_weight = np.sum(sample_weights[right_idx]) / np.sum(sample_weights)

        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
    
    def _entropy(self, y, sample_weights):
        class_counts = {}
        # Calculating the weighted counts of the classes
        for label in np.unique(y):
            class_counts[label] = np.sum(sample_weights[np.where(y == label)])

        total_weight = np.sum(sample_weights)
        entropy = 0
        for label in class_counts:
            prob = class_counts[label] / total_weight
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def _traverse_tree(self, sample, node):
        if "leaf" in node:
            return node["leaf"]
        if sample[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(sample, node["left"])
        return self._traverse_tree(sample, node["right"])
    
if __name__ == "__main__":
    print("Decision Tree module loaded successfully!")