"""
Programmer: Murat
Class: CPSC 322, Fall 2024
Programming Assignment: 3
Date: 11/7/2024
I attempted the bonus: Yes


"""
from collections import defaultdict, Counter
import random
import math
import numpy as np
from graphviz import Digraph
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

def discretizer(y_value):
    """ The discretizer() function is a simple utility function that takes a
    numeric input y_value and converts it into one of two categorical labels: high or low.
    """
    if y_value >= 100:
        return "high"
    return "low"

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer_fn, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer_fn
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred_numeric = self.regressor.predict(X_test)
        y_pred_discrete = []
        for y in y_pred_numeric:
            y_pred_discrete.append(self.discretizer(y))
        return y_pred_discrete

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance."""
        neighbor_indices = []
        distances = []
        for test_idx, test_instance in enumerate(X_test):
            current_distances = []
            for i, train_instance in enumerate(self.X_train):
                try:
                    test_instance = [float(x) for x in test_instance]
                    train_instance = [float(x) for x in train_instance]

                    dist = np.linalg.norm(np.array(test_instance) - np.array(train_instance))
                    current_distances.append((dist, i))
                except ValueError as ve:
                    print(f"Ошибка преобразования данных в число:")
                    print(f"Test instance: {test_instance}")
                    print(f"Train instance: {train_instance}")
                    print(f"Ошибка: {ve}")
                    raise ve
                except Exception as e:
                    print(f"Неизвестная ошибка на тестовом элементе {test_idx} и тренировочном элементе {i}: {e}")
                    raise e
            current_distances.sort(key=lambda x: x[0])
            k_nearest = current_distances[:self.n_neighbors]
            neighbor_indices.append([index for _, index in k_nearest])
            distances.append([dist for dist, _ in k_nearest])
        return distances, neighbor_indices


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for test_instance in X_test:
            _, neighbor_indices = self.kneighbors([test_instance])
            neighbor_labels = [self.y_train[i] for i in neighbor_indices[0]]
            label_counts = {}
            for label in neighbor_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            most_common_label = None
            max_count = -1
            for label, count in label_counts.items():
                if count > max_count:
                    max_count = count
                    most_common_label = label
            y_pred.append(most_common_label)

        return y_pred

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self, strategy="most_frequent"):
        self.strategy = strategy
        self.most_common_label = None
        self.class_probabilities = {}

    def fit(self, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        label_counts = {}
        total_count = len(y_train)

        # Counting occurrences of each class in y_train
        for label in y_train:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # "most_frequent" strategy: find the most frequent class
        if self.strategy == "most_frequent":
            most_common_count = -1
            for label, count in label_counts.items():
                if count > most_common_count:
                    most_common_count = count
                    self.most_common_label = label

        # "stratified" strategy: calculate probabilities for each class
        elif self.strategy == "stratified":
            for label, count in label_counts.items():
                self.class_probabilities[label] = count / total_count

    def predict(self, X_test):
        """Makes predictions for test instances based on the selected strategy.

        Args:
            X_test (list of list of obj): The test instances to classify.

        Returns:
            y_pred (list of obj): The predicted target values.
        """
        y_pred = []

        if self.strategy == "most_frequent":
            # Predict the most frequent class for every instance
            y_pred = [self.most_common_label] * len(X_test)

        elif self.strategy == "stratified":
            # Extract classes and their probabilities manually
            classes = list(self.class_probabilities.keys())
            probabilities = list(self.class_probabilities.values())

            # Predict randomly based on the class probabilities
            for _ in X_test:
                y_pred.append(random.choices(classes, weights=probabilities, k=1)[0])

        return y_pred


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict): The prior probabilities computed for each
            label in the training set.
        posteriors(dict): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        class_counts = Counter(y_train)
        total_count = len(y_train)
        self.priors = {}
        for class_label in class_counts:
            self.priors[class_label] = class_counts[class_label] / total_count

        n_features = len(X_train[0])
        self.posteriors = {}

        for class_label in class_counts:
            indices = [i for i, y in enumerate(y_train) if y == class_label]

            feature_value_counts = [defaultdict(int) for _ in range(n_features)]
            total_feature_counts = [0] * n_features

            for idx in indices:
                instance = X_train[idx]
                for i in range(n_features):
                    value = instance[i]
                    feature_value_counts[i][value] += 1
                    total_feature_counts[i] += 1

            feature_value_probs = []
            for i in range(n_features):
                total = total_feature_counts[i]
                probs = {}
                for value in feature_value_counts[i]:
                    probs[value] = feature_value_counts[i][value] / total
                feature_value_probs.append(probs)

            self.posteriors[class_label] = feature_value_probs

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            class_probabilities = {}
            for class_label in self.priors:
                probability = self.priors[class_label]
                for i, feature_value in enumerate(instance):
                    feature_probs = self.posteriors[class_label][i]
                    p_feature_given_class = feature_probs.get(feature_value, 0)
                    if p_feature_given_class == 0:
                        probability = 0
                        break
                    probability *= p_feature_given_class
                class_probabilities[class_label] = probability
            # Choose the class_label with the highest probability
            max_prob = max(class_probabilities.values())
            # Handle the case where all probabilities are zero
            if max_prob == 0:
                y_predicted.append(None)
            else:
                # Get class_labels with max probability (in case of tie)
                predicted_classes = [class_label for class_label, prob in class_probabilities.items() if prob == max_prob]
                # If tie, pick the first one (or decide how to handle ties)
                y_predicted.append(predicted_classes[0])
        return y_predicted


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        data = [x + [y] for x, y in zip(X_train, y_train)]
        self.tree = self._treebuilder(data, list(range(len(X_train[0]))))

    def _treebuilder(self, data, available_attributes, parent_size=None):
        labels = [row[-1] for row in data]

        if len(set(labels)) == 1:
            return ["Leaf", labels[0], len(data), parent_size]

        if not available_attributes:
            majority_label = self._majority_vote(labels)
            return ["Leaf", majority_label, len(data), parent_size]

        best_attr = self._select_best_attribute(data, available_attributes)
        tree = ["Attribute", f"att{best_attr}"]

        # Debugging: Проверить содержимое столбца
        try:
            attr_values = sorted(set(row[best_attr] for row in data))
        except TypeError as e:
            print(f"Ошибка в столбце: {best_attr}")
            print(f"Значения в столбце: {[row[best_attr] for row in data]}")
            raise e

        available_attributes.remove(best_attr)

        for value in attr_values:
            subset = [row for row in data if row[best_attr] == value]
            if not subset:
                majority_label = self._majority_vote(labels)
                tree.append(["Value", value, ["Leaf", majority_label, 0, len(data)]])
            else:
                subtree = self._treebuilder(subset, available_attributes[:], len(data))
                tree.append(["Value", value, subtree])

        return tree

    def _majority_vote(self, labels):
            """Returns the majority label from a list of labels."""
            label_counts = Counter(labels)
            majority_label = None
            max_count = -1

            for label, count in label_counts.items():
                if count > max_count or (count == max_count and label < majority_label):
                    majority_label = label
                    max_count = count

            return majority_label


    def _select_best_attribute(self, data, attributes):
        """Selects the attribute with the highest information gain."""
        base_entropy = self._entropy([row[-1] for row in data])
        max_info_gain = -1
        best_attr = None

        for attr in attributes:
            attr_values = set(row[attr] for row in data)
            weighted_entropy = 0

            for value in attr_values:
                subset = [row for row in data if row[attr] == value]
                subset_labels = [row[-1] for row in subset]
                weighted_entropy += (len(subset) / len(data)) * self._entropy(subset_labels)

            info_gain = base_entropy - weighted_entropy

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attr = attr

        return best_attr

    def _entropy(self, labels):
        """Calculates the entropy of a list of labels."""
        label_counts = Counter(labels)
        total = len(labels)
        entropy = 0

        for count in label_counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)

        return entropy


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        for instance in X_test:
            prediction = self._traverse_tree(self.tree, instance)
            y_predicted.append(prediction)

        return y_predicted

    def _traverse_tree(self, tree, instance):
        """Recursively traverses the decision tree to make a prediction."""
        if tree[0] == "Leaf":
            return tree[1]  # Return the predicted label
        if tree[0] == "Attribute":
            attr_index = int(tree[1][3:])  # Extract attribute index (e.g., "att0" -> 0)
            attr_value = instance[attr_index]
            for branch in tree[2:]:  # Look through the branches
                if branch[1] == attr_value:  # If the value matches the branch
                    return self._traverse_tree(branch[2], instance)  # Recursively traverse the subtree
        return None  # Return None if no match is found (shouldn't happen in a valid tree)

    def print_decision_rules(self, attribute_names=None, class_name="class", decode_map=None):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
            decode_map(dict or None): A dictionary to decode attribute and class values (optional).
        """
        def traverse_tree(node, conditions):
            """Recursively traverse the tree and collect rules."""
            if node is None:  # Check for None
                return
            if node[0] == "Leaf":
                rule = " AND ".join(conditions)
                class_label = decode_map.get(node[1], node[1]) if decode_map else node[1]
                print(f"IF {rule} THEN {class_name} = {class_label}")
            elif node[0] == "Attribute":
                attr_name = attribute_names[int(node[1][3:])] if attribute_names else node[1]
                for branch in node[2:]:
                    branch_value = decode_map.get(branch[1], branch[1]) if decode_map else branch[1]
                    traverse_tree(branch[2], conditions + [f"{attr_name} == {branch_value}"])

        # Start the recursive traversal from the root of the tree
        if self.tree is not None:  # Ensure the tree is not None
            traverse_tree(self.tree, [])
        else:
            print("The decision tree is empty or not initialized.")


        # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
        """
        def add_nodes_edges(dot, node, node_id):
            """Recursively adds nodes and edges to the graph."""
            if node[0] == "Leaf":
                # Add a leaf node
                dot.node(
                    str(node_id),
                    f"Leaf: {node[1]}\n({node[2]}/{node[3]})",
                    shape="box"
                )
            elif node[0] == "Attribute":
                # Create an attribute node
                attr_name = attribute_names[int(node[1][3:])] if attribute_names else node[1]
                dot.node(str(node_id), f"{attr_name}")
                for i, branch in enumerate(node[2:]):
                    # Add an edge and recurse for child nodes
                    child_id = f"{node_id}.{i}"  # Create a unique child ID
                    dot.edge(str(node_id), child_id, label=str(branch[1]))
                    add_nodes_edges(dot, branch[2], child_id)

        # Initialize Graphviz Digraph
        dot = Digraph(format="pdf")
        dot.attr(dpi="300")

        # Start building the tree from the root
        if self.tree is not None:
            add_nodes_edges(dot, self.tree, "0")
        else:
            print("The decision tree is empty or not initialized.")

        # Save the graph to .dot and .pdf
        dot.render(filename=dot_fname, cleanup=True)
        print(f"Tree visual saved to {dot_fname}.dot and {pdf_fname}.pdf")

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        N (int): The number of trees in the forest.
        M (int): The minimum number of samples required to split an internal node.
        F (int): The number of features to consider when looking for the best split.
        decision_trees (list): The list of decision trees in the forest.
        bootstraps_datasets (list): The list of bootstrap datasets for each tree.
        bootstraps_labels (list): The list of labels for each bootstrap dataset.
    """

    def __init__(self, N=20, M=7, F=2):
        """Initializer for MyRandomForestClassifier.

        Args:
            N (int): The number of trees in the forest.
            M (int): The minimum number of samples required to split an internal node.
            F (int): The number of features to consider when looking for the best split.
        """
        self.N = N
        self.M = M
        self.F = F
        self.decision_trees = []
        self.bootstraps_datasets = []
        self.bootstraps_labels = []
        self.y_train = None  # Добавлено для сохранения обучающих меток

    def fit(self, X_train, y_train):
        """Fits the random forest classifier to the training data.

        Args:
            X_train (list of list of obj): The training data features.
            y_train (list of obj): The training data labels.
        """
        self.X_train = X_train  # Сохраняем обучающие данные (по желанию)
        self.y_train = y_train  # Сохраняем обучающие метки
        self.decision_trees = []
        self.bootstraps_datasets = []
        self.bootstraps_labels = []
        for _ in range(self.N):
            bootstrap_data, bootstrap_labels = self._bootstrapping(X_train, y_train)
            self.bootstraps_datasets.append(bootstrap_data)
            self.bootstraps_labels.append(bootstrap_labels)
            features_indices = random.sample(range(len(X_train[0])), self.F)
            bootstrap_data_sub = [[row[i] for i in features_indices] for row in bootstrap_data]
            tree = MyDecisionTreeClassifier()
            tree.fit(bootstrap_data_sub, bootstrap_labels)
            tree.selected_features = features_indices
            self.decision_trees.append(tree)

    def predict(self, X_test):
        """Predicts the labels for the test data.

        Args:
            X_test (list of list of obj): The test data features.

        Returns:
            y_predicted (list of obj): The predicted labels.
        """
        predictions = []
        for tree in self.decision_trees:
            X_test_sub = [[row[i] for i in tree.selected_features] for row in X_test]
            y_pred = tree.predict(X_test_sub)
            predictions.append(y_pred)
        y_predicted = self._majority_vote(predictions)
        return y_predicted

    def _bootstrapping(self, X_train, y_train):
        """Creates a bootstrap sample from the training data.

        Args:
            X_train (list of list of obj): The training data features.
            y_train (list of obj): The training data labels.

        Returns:
            bootstrap_dataset (list of list of obj): The bootstrap sample features.
            bootstrap_labels (list of obj): The bootstrap sample labels.
        """
        n_samples = len(X_train)
        bootstrap_dataset = []
        bootstrap_labels = []
        for _ in range(n_samples):
            index = random.randint(0, n_samples - 1)
            bootstrap_dataset.append(X_train[index])
            bootstrap_labels.append(y_train[index])
        return bootstrap_dataset, bootstrap_labels

    def _majority_vote(self, predictions):
        """Aggregates the predictions from multiple trees using majority vote.

        Args:
            predictions (list of list of obj): The list of predictions from each tree.

        Returns:
            final_predictions (list of obj): The aggregated predictions.
        """
        final_predictions = []
        n_samples = len(predictions[0])
        for i in range(n_samples):
            votes = [pred[i] for pred in predictions if pred[i] is not None]
            if votes:
                majority_vote = Counter(votes).most_common(1)[0][0]
            else:
                # Если все предсказания None, используем наиболее частый класс из обучающих меток
                majority_vote = Counter(self.y_train).most_common(1)[0][0]
            final_predictions.append(majority_vote)
        return final_predictions
