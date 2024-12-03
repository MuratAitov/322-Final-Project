"""
Programmer: Murat
Class: CPSC 322, Fall 2024
Programming Assignment: 3
Date: 11/7/2024
I attempted the bonus: Yes


"""


import math
import numpy as np # use numpy's random number generation
import tabulate

from mysklearn import myclassifiers


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """


    data = list(zip(X, y))

    if random_state:
        np.random.seed(random_state)
    if shuffle:
        np.random.shuffle(data)
    if isinstance(test_size, float):
        test_size = math.ceil(len(X) * test_size)

    # Ensure test size does not exceed the number of samples
    test_size = min(test_size, len(X))

    # Unzip data
    X, y = zip(*data)
    # Calculate the number of training samples
    n_train = len(X) - test_size
    # Split data into training and testing sets
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]


    return list(X_train), list(X_test), list(y_train), list(y_test)


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    n_samples = len(X)
    indices = np.arange(n_samples)

    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    folds = []
    fold_sizes = [(n_samples // n_splits + 1) if i < n_samples % n_splits else (n_samples // n_splits)
                  for i in range(n_splits)]

    current = 0
    for fold_size in fold_sizes:
        test_indices = indices[current:current + fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        folds.append((train_indices.tolist(), test_indices.tolist()))
        current += fold_size

    return folds


# BONUS function
def stratified_kfold_split(_, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random

    y = np.array(y)
    indices = np.arange(len(y))

    if shuffle:
        indices = indices.copy()
        rng.shuffle(indices)

    class_indices = {}
    for idx in indices:
        label = y[idx]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    folds = [[] for _ in range(n_splits)]
    for label, idxs in class_indices.items():
        for i, idx in enumerate(idxs):
            folds[i % n_splits].append(idx)

    stratified_folds = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = np.setdiff1d(indices, test_indices)
        stratified_folds.append((train_indices.tolist(), test_indices))

    return stratified_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """

    if n_samples is None:
        n_samples = len(X)

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.randint(0, len(X), n_samples)
    X_sample = [X[i] for i in indices]
    y_sample = [y[i] for i in indices] if y is not None else None

    out_of_bag_indices = set(range(len(X))) - set(indices)
    X_out_of_bag = [X[i] for i in out_of_bag_indices]
    y_out_of_bag = [y[i] for i in out_of_bag_indices] if y is not None else None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0] * len(labels) for _ in labels]

    for true, pred in zip(y_true, y_pred):
        true_index = labels.index(true)
        pred_index = labels.index(pred)
        matrix[true_index][pred_index] += 1

    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    if normalize:
        return correct / len(y_true)
    return correct


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    # Count true positives (tp) and false positives (fp)
    tp = sum((y_t == pos_label and y_p == pos_label) for y_t, y_p in zip(y_true, y_pred))
    fp = sum((y_t != pos_label and y_p == pos_label) for y_t, y_p in zip(y_true, y_pred))

    # Avoid division by zero
    if tp + fp == 0:
        return 0.0

    precision = tp / (tp + fp)
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    # Count true positives (tp) and false negatives (fn)
    tp = sum((y_t == pos_label and y_p == pos_label) for y_t, y_p in zip(y_true, y_pred))
    fn = sum((y_t == pos_label and y_p != pos_label) for y_t, y_p in zip(y_true, y_pred))

    # Avoid division by zero
    if tp + fn == 0:
        return 0.0

    recall = tp / (tp + fn)
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    # Avoid division by zero
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def classification_report(y_true, y_pred, labels=None, output_dict=False):
    """Build a text report and a dictionary showing the main classification metrics.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        output_dict(bool): If True, return output as dict instead of a str

    Returns:
        report(str or dict): Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the following structure:
                {'label 1': {'precision': 0.5,
                             'recall': 1.0,
                             'f1-score': 0.67,
                             'support': 1},
                 'label 2': { ... },
                 ... }
            Includes macro and weighted averages for precision, recall, and F1-score.
    """
    if labels is None:
        labels = sorted(list(set(y_true)))

    report_dict = {}
    total_support = len(y_true)

    for label in labels:
        precision = binary_precision_score(y_true, y_pred, pos_label=label)
        recall = binary_recall_score(y_true, y_pred, pos_label=label)
        f1 = binary_f1_score(y_true, y_pred, pos_label=label)
        support = sum(1 for y in y_true if y == label)

        report_dict[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }

    # Compute macro and weighted averages
    macro_avg = {
        'precision': np.mean([report_dict[label]['precision'] for label in labels]),
        'recall': np.mean([report_dict[label]['recall'] for label in labels]),
        'f1-score': np.mean([report_dict[label]['f1-score'] for label in labels]),
        'support': total_support
    }

    weighted_avg = {
        'precision': np.sum([report_dict[label]['precision'] * report_dict[label]['support'] for label in labels]) / total_support,
        'recall': np.sum([report_dict[label]['recall'] * report_dict[label]['support'] for label in labels]) / total_support,
        'f1-score': np.sum([report_dict[label]['f1-score'] * report_dict[label]['support'] for label in labels]) / total_support,
        'support': total_support
    }

    report_dict['macro avg'] = macro_avg
    report_dict['weighted avg'] = weighted_avg

    if output_dict:
        return report_dict

    # Convert to tabulate format
    table = [[
        label,
        report_dict[label]['precision'],
        report_dict[label]['recall'],
        report_dict[label]['f1-score'],
        report_dict[label]['support']
    ] for label in labels] + [
        ['macro avg', macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score'], macro_avg['support']],
        ['weighted avg', weighted_avg['precision'], weighted_avg['recall'], weighted_avg['f1-score'], weighted_avg['support']]
    ]

    return tabulate.tabulate(table, headers=['Class', 'Precision', 'Recall', 'F1-score', 'Support'], floatfmt=".2f")

def cross_validate(classifier, X, y, folds, scoring=None):
    """Perform cross-validation on a classifier.

    Args:
        classifier: The classifier instance (must have fit() and predict() methods).
        X (list of list): The dataset features.
        y (list): The target labels.
        folds (list of tuples): The train-test split indices for cross-validation.
        scoring (function or None): Scoring function (e.g., accuracy_score). If None, defaults to accuracy.

    Returns:
        dict: A dictionary with metrics such as accuracy, precision, recall, and F1-score.
    """
    from collections import defaultdict
    metrics = defaultdict(list)

    for train_idx, test_idx in folds:
        # Split the dataset into training and testing sets
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        # Train the classifier
        if isinstance(classifier, myclassifiers.MyDummyClassifier):
            classifier.fit(y_train)  # Только y_train
        else:
            classifier.fit(X_train, y_train)  # X_train и y_train для других классификаторов


        # Predict the target labels
        y_pred = classifier.predict(X_test)

        # Calculate metrics
        if scoring is None:
            scoring = accuracy_score  # Default to accuracy
        metrics['accuracy'].append(scoring(y_test, y_pred))
        metrics['precision'].append(binary_precision_score(y_test, y_pred))
        metrics['recall'].append(binary_recall_score(y_test, y_pred))
        metrics['f1'].append(binary_f1_score(y_test, y_pred))

    # Aggregate metrics (average scores)
    results = {metric: np.mean(scores) for metric, scores in metrics.items()}
    return results
