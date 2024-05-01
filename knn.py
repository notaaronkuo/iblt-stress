from scipy.spatial.distance import euclidean


class StressPredictorKNN:

    def __init__(self, k):
        """
        Standard kNN algorithm, modified to store frequency of nearest-neighbors used.
        Can evict the least x used neighbors.
        :param k: k of kNN
        """
        self.k = k
        self.emotion_matrix = []
        self.X = []
        self.y = []
        self.sample_frequency = []

    def fit(self, X, y):
        """
        Train the KNN model.

        :param X: Array-like, shape (n_samples, n_features) Training data, where n_samples is the number of samples and n_features is the number of features.
        :param y: Array-like, shape (n_samples,) Target values.
        """
        self.X = X
        self.y = y

    def append_sample(self, X, y):
        """
        Add sample(s) to dataset
        :param X: Text(s), appendable to array, index-matched to y
        :param y: Stress value(s), appendable to array, index-matched to X
        :return: None
        """
        self.X.append(X)
        self.y.append(y)

    def can_predict(self):
        """
        Check if the model can make predictions.

        :return: True if the model is trained with sufficient size, False otherwise.
        """
        return len(self.X) >= self.k

    def predict(self, x_test):
        """
        Predict the stress level for a single test sample. Uses Euclidean distance.

        :param x_test: Array-like, shape (n_features,) Test sample.
        :return: Predicted stress level.
        """
        if not self.can_predict():
            raise RuntimeError("KNN: Model has not been trained yet. Call fit() before predict().")

        distances = []
        for i, x_train in enumerate(self.X):
            dist = euclidean(x_test, x_train)
            distances.append((dist, self.y[i], i))  # Add index i of training sample
        distances.sort()
        k_nearest = distances[:self.k]
        stress_levels = [level for _, level, _ in k_nearest]
        prediction = max(set(stress_levels), key=stress_levels.count)
        nearest_neighbors_indices = [index for _, _, index in k_nearest]
        self.sample_frequency.extend(nearest_neighbors_indices)
        # return prediction, nearest_neighbors_indices
        return prediction

    def evict_LFU(self, n=0):
        """
        Using methodology used in cache management, evicts the Least Frequently Used nearest-neighbors.
        Removes 30% of N by default.
        :param n: number of LFU neighbors to remove.
        :return: None
        """
        if n == 0:
            n = round(len(self.y) * 0.30)
        elif n >= len(self.y):
            raise RuntimeError("KNN: Cannot remove greater than N neighbors!")

        # Create a set of numbers in the given list
        numbers_set = set(self.sample_frequency)

        # Generate a list of numbers not present in the given list up to len(self.X)
        numbers_not_in_list = [number for number in range(len(self.X)) if number not in numbers_set]

        # Sort the numbers from the given list based on the number of occurrences
        sorted_numbers = sorted(self.sample_frequency, key=lambda x: self.sample_frequency.count(x), reverse=True)

        # Concatenate the two lists
        result_list = numbers_not_in_list + sorted_numbers

        dropped_indices = set(result_list[:n])
        del result_list[:n]
        self.sample_frequency = result_list
        for i in reversed(sorted(dropped_indices)):
            del self.X[i]
            del self.y[i]


def main():
    """
    For testing purposes, can be deleted
    :return: None
    """
    # Sample data
    X_train = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,
         0.5, 0.6, 0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5,
         0.6, 0.7, 0.8, 0.9, 0.1],
        [0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
         0.9,
         0.1, 0.2, 0.3, 0.4],
        [0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
         0.1,
         0.2, 0.3, 0.4, 0.5]
        # Add more training samples as needed
    ]
    y_train = [[1, 2], [3, 1]]  # Corresponding stress levels for the training data

    X_test = [
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
         0.7, 0.8, 0.9, 0.1, 0.2],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
         0.8, 0.9, 0.1, 0.2, 0.3],
        [0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1,
         0.2,
         0.3, 0.4, 0.5, 0.6],
        [0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2,
         0.3,
         0.4, 0.5, 0.6, 0.7]
        # Add more test samples as needed
    ]

    # Expected output after nearest neighbor calculation
    # Assuming k=3, let's say for the first test sample, the nearest neighbors are indices 0, 1, and 2 in the training data.
    # And for the second test sample, the nearest neighbors are indices 1, 2, and 3 in the training data.
    # We'll create a list accordingly.

    nearest_neighbors_indices_expected = [[0, 1, 2],
                                          [1, 2, 3]]  # Expected nearest neighbors indices for each test sample

    # Initialize and train the KNN model
    knn = StressPredictorKNN(k=3)  # Initialize with k=3
    knn.fit(X_train, y_train)

    # Predict stress levels for test data
    predictions, _ = knn.predict(X_test[0])
    print("Predicted stress levels:", predictions)


if __name__ == "__main__":
    main()
