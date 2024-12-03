from mysklearn.myclassifiers import MyRandomForestClassifier
import unittest
import random
import math
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class TestMyRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        self.X = iris.data.tolist()
        self.y = iris.target.tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
        self.clf = MyRandomForestClassifier(N=10, M=2, F=2)

    def test_init(self):
        self.assertEqual(self.clf.N, 10)
        self.assertEqual(self.clf.M, 2)
        self.assertEqual(self.clf.F, 2)
        self.assertEqual(self.clf.decision_trees, [])
        self.assertEqual(self.clf.bootstraps_datasets, [])
        self.assertEqual(self.clf.bootstraps_labels, [])

    def test_fit(self):
        self.clf.fit(self.X_train, self.y_train)
        self.assertEqual(len(self.clf.decision_trees), self.clf.N)
        self.assertEqual(len(self.clf.bootstraps_datasets), self.clf.N)
        self.assertEqual(len(self.clf.bootstraps_labels), self.clf.N)
        for tree in self.clf.decision_trees:
            self.assertIsNotNone(tree.tree)

    def test_predict(self):
            self.clf.fit(self.X_train, self.y_train)
            predictions = self.clf.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            self.assertFalse(any(pred is None for pred in predictions))

    def test_bootstrapping(self):
        dataset, labels = self.clf._bootstrapping(self.X_train, self.y_train)
        self.assertEqual(len(dataset), len(self.X_train))
        self.assertEqual(len(labels), len(self.y_train))
        self.assertNotEqual(dataset, self.X_train)

    def test_majority_vote(self):
        predictions = []
        for _ in range(self.clf.N):
            preds = [random.choice(self.y_train) for _ in range(len(self.X_test))]
            predictions.append(preds)
        final_predictions = self.clf._majority_vote(predictions)
        self.assertEqual(len(final_predictions), len(self.X_test))
        self.assertFalse(any(pred is None for pred in final_predictions))

    def test_full_pipeline(self):
        self.clf.fit(self.X_train, self.y_train)
        predictions = self.clf.predict(self.X_test)
        correct = sum(1 for true, pred in zip(self.y_test, predictions) if true == pred)
        accuracy = correct / len(self.y_test)
        print(f"Accuracy: {accuracy}")
        self.assertGreaterEqual(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()