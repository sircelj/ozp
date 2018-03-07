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

        for my_pca in (PowerPCA, OrtoPCA):
            my_pca = my_pca(n_components=n_comp)
            my_pca.fit(self.X)

            self.assertEqual(my_pca.explained_variance_.shape, (3, ))
            np.testing.assert_array_almost_equal(my_pca.explained_variance_,
                                                 s_pca.explained_variance_, decimal=1)

            diff = sum(min(np.linalg.norm(r1 - r2), np.linalg.norm(r1 + r2))
                       for r1, r2 in zip(my_pca.components_, s_pca.components_))
            self.assertAlmostEqual(diff, 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
