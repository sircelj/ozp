import unittest
import sklearn
import numpy as np
from sklearn import datasets

from pca import PowerPCA, OrtoPCA, EigenPCA


class PCATest(unittest.TestCase):
    def setUp(self):
        X = datasets.load_iris().data
        self.X = X - X.mean(axis=0)
        self.n_comp = 3

    def test_pca(self):
        s_pca = sklearn.decomposition.PCA(self.n_comp)
        s_pca.fit(self.X)

        for my_pca in (PowerPCA, OrtoPCA, EigenPCA):
            my_pca = my_pca(n_components=self.n_comp)
            my_pca.fit(self.X)

            self.assertEqual(my_pca.explained_variance_.shape, (self.n_comp, ))
            np.testing.assert_array_almost_equal(my_pca.explained_variance_,
                                                 s_pca.explained_variance_, decimal=1)
            np.testing.assert_array_almost_equal(my_pca.explained_variance_ratio_,
                                                 s_pca.explained_variance_ratio_, decimal=1)

            diff = sum(min(np.linalg.norm(r1 - r2), np.linalg.norm(r1 + r2))
                       for r1, r2 in zip(my_pca.components_, s_pca.components_))
            self.assertAlmostEqual(diff, 0.)

    def test_transform(self):
        n_sample = 5
        train, test = self.X[:-n_sample], self.X[-n_sample:]
        s_pca = sklearn.decomposition.PCA(self.n_comp)
        s_pca.fit(train)
        sk_T = s_pca.transform(test)

        for my_pca in (PowerPCA, OrtoPCA, EigenPCA):
            my_pca = my_pca(self.n_comp)
            my_pca.fit(train)

            my_T = my_pca.transform(self.X[-n_sample:])
            self.assertEqual(my_T.shape, (n_sample, self.n_comp))
            for c1, c2 in zip(sk_T.T, my_T.T):
                np.testing.assert_almost_equal(min(np.linalg.norm(c1 - c2), np.linalg.norm(c1 + c2)), 0.)

if __name__ == "__main__":
    unittest.main(verbosity=3)
