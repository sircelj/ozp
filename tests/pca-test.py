import unittest
import sklearn.decomposition
import numpy as np
from sklearn import datasets

from pca import PowerPCA, OrtoPCA


class PCATest(unittest.TestCase):
    def setUp(self):
        X = datasets.load_iris().data
        self.X = X - X.mean(axis=0)

    def test_pca(self):
        n_comp = 3
        s_pca = sklearn.decomposition.PCA(n_comp)
        s_pca.fit(self.X)

        s_pca_X = s_pca.transform(self.X)

        for my_pca in (PowerPCA, OrtoPCA):
            my_pca = my_pca(n_components=n_comp)
            my_pca.fit(self.X)

            my_pca_X = my_pca.transform(self.X)

            self.assertEqual(my_pca.explained_variance_.shape, (3, ))
            np.testing.assert_array_almost_equal(my_pca.explained_variance_,
                                                 s_pca.explained_variance_, decimal=5)

            np.testing.assert_array_almost_equal(my_pca.explained_variance_ratio_,
                                                 s_pca.explained_variance_ratio_, decimal=5)

            diff = sum(min(np.linalg.norm(r1 - r2), np.linalg.norm(r1 + r2))
                       for r1, r2 in zip(my_pca.components_, s_pca.components_))
            self.assertAlmostEqual(diff, 0.)

            # Check if the components_ are paralel and of the same length
            for r1, r2 in zip(my_pca.components_, s_pca.components_):
                self.assertAlmostEqual(abs(r1.dot(r2)) / np.linalg.norm(r1) ** 2, 1.)
                self.assertAlmostEqual(np.linalg.norm(r1), np.linalg.norm(r2))

            # Check if the transforms are paralel and of the same length
            for j in range(n_comp):
                self.assertAlmostEqual(abs(my_pca_X[:, j].dot(s_pca_X[:, j])) / np.linalg.norm(my_pca_X[:, j]) ** 2, 1.)
                self.assertAlmostEqual(np.linalg.norm(my_pca_X[:, j]), np.linalg.norm(my_pca_X[:, j]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
