import unittest
import numpy as np
from sklearn import datasets

from ann import NeuralNetwork


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        self.X, self.y = iris.data, iris.target

    def test_set_data(self):
        ann = NeuralNetwork([5, 3], alpha=0.01)
        ann.set_data_(self.X, self.y)
        coefs = ann.init_weights_()
        self.assertEqual(len(coefs), 55)

    def test_weighs_structure(self):
        ann = NeuralNetwork([5, 3], alpha=0.01)
        ann.set_data_(self.X, self.y)
        coefs = ann.unflatten_coefs(ann.init_weights_())
        shapes = np.array([coef.shape for coef in coefs])
        np.testing.assert_array_equal(shapes, np.array([[5, 5], [6, 3], [4, 3]]))

    def test_gradient_computation(self):
        ann = NeuralNetwork([2, 2], alpha=0.01)
        ann.set_data_(self.X, self.y)
        coefs = ann.init_weights_()
        g1 = ann.grad_approx(coefs, e=1e-5)
        g2 = ann.grad(coefs)
        np.testing.assert_array_almost_equal(g1, g2, decimal=10)

    def test_fit_and_predict(self):
        ann = NeuralNetwork([4, 2], alpha=0.01)
        ann.fit(self.X, self.y)
        T = self.X[[10, 60, 110]]
        predictions = ann.predict(T)
        np.testing.assert_array_equal(predictions, np.array([0, 1, 2]))

    def test_predict_probabilities(self):
        ann = NeuralNetwork([4, 2], alpha=0.01)
        ann.fit(self.X, self.y)
        T = self.X[[15, 65, 115, 117]]
        ps = ann.predict_proba(T)
        margin = np.min(np.max(ps, axis=1))
        self.assertGreater(margin, 0.90)


if __name__ == "__main__":
    unittest.main(verbosity=2)
