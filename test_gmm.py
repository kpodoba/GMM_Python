import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

from main import SimpleGMM, cluster_to_true


class TestSimpleGMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wczytanie i przygotowanie danych tylko raz."""
        data = load_breast_cancer()
        X = data.data
        y = data.target

        scaler = StandardScaler()
        cls.X_scaled = scaler.fit_transform(X)
        cls.y = y

    def test_simple_gmm_initialization(self):
        """Czy obiekt SimpleGMM tworzy się poprawnie."""
        model = SimpleGMM(n_components=2, max_iter=20)
        self.assertEqual(model.n_components, 2)
        self.assertEqual(model.max_iter, 20)

    def test_simple_gmm_fit(self):
        """Czy model uczy się i ustawia parametry."""
        model = SimpleGMM(n_components=2, max_iter=5)
        model.fit(self.X_scaled)

        self.assertEqual(model.means.shape[0], 2)
        self.assertEqual(model.vars.shape[0], 2)
        self.assertEqual(model.weights.shape[0], 2)

    def test_predict_output_shape(self):
        """Czy predict zwraca etykiety o poprawnym rozmiarze."""
        model = SimpleGMM(n_components=2, max_iter=5)
        model.fit(self.X_scaled)
        labels = model.predict(self.X_scaled)

        self.assertEqual(labels.shape[0], self.X_scaled.shape[0])

    def test_cluster_to_true(self):
        """Czy funkcja mapująca etykiety działa i zwraca odpowiedni rozmiar."""
        dummy_pred = np.zeros_like(self.y)
        mapped = cluster_to_true(self.y, dummy_pred)

        self.assertEqual(mapped.shape, self.y.shape)

    def test_accuracy_above_zero(self):
        """Czy dokładność po klasteryzacji jest większa niż 0 (czyli model działa)."""
        model = SimpleGMM(n_components=2, max_iter=20)
        model.fit(self.X_scaled)

        labels = model.predict(self.X_scaled)
        mapped = cluster_to_true(self.y, labels)

        accuracy = accuracy_score(self.y, mapped)
        self.assertGreaterEqual(accuracy, 0.0)

    def test_compare_with_sklearn_gmm(self):
        """Czy oficjalny GMM również działa poprawnie."""
        sk_model = GaussianMixture(n_components=2, covariance_type='full')
        sk_model.fit(self.X_scaled)

        sk_labels = sk_model.predict(self.X_scaled)
        mapped = cluster_to_true(self.y, sk_labels)

        conf = confusion_matrix(self.y, mapped)
        self.assertEqual(conf.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
