import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use('Agg')


# 1. IMPLEMENTACJA WŁASNEGO GMM (SimpleGMM)
class SimpleGMM:
    def __init__(self, n_components=2, max_iter=50, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        np.random.seed(random_state)

    def gaussian(self, X, mean, var):
        return np.exp(-0.5 * np.sum((X - mean)**2 / var, axis=1)) / \
               np.sqrt(np.prod(2 * np.pi * var))

    def fit(self, X):
        # wymiar danych
        n, d = X.shape

        # inicjalizacja parametrów
        self.means = X[np.random.choice(n, self.n_components, replace=False)]
        self.vars = np.ones((self.n_components, d))
        self.weights = np.ones(self.n_components) / self.n_components

        # pętla EM
        for _ in range(self.max_iter):

            # E-step
            resp = np.zeros((n, self.n_components))
            for k in range(self.n_components):
                resp[:, k] = self.weights[k] * \
                             self.gaussian(X, self.means[k], self.vars[k])
            resp = resp / np.sum(resp, axis=1, keepdims=True)

            # M-step
            Nk = np.sum(resp, axis=0)

            for k in range(self.n_components):
                self.means[k] = (resp[:, k].reshape(-1, 1) * X).sum(axis=0) / Nk[k]

                diff = X - self.means[k]
                self.vars[k] = (resp[:, k].reshape(-1, 1) *
                                (diff ** 2)).sum(axis=0) / Nk[k]

            self.weights = Nk / n

    def predict(self, X):
        n = X.shape[0]
        resp = np.zeros((n, self.n_components))

        for k in range(self.n_components):
            resp[:, k] = self.weights[k] * \
                         self.gaussian(X, self.means[k], self.vars[k])

        return np.argmax(resp, axis=1)

# 2. METODY WSPÓLNE DLA OBU GMM (mapowanie, przygotowanie danych)
def cluster_to_true(y_true, y_pred, n_components=2):
    mapped_labels = np.zeros_like(y_pred)
    for cluster_id in range(n_components):
        mask = (y_pred == cluster_id)
        if mask.sum() == 0:
            continue
        majority_label = mode(y_true[mask], keepdims=True)[0]
        mapped_labels[mask] = majority_label
    return mapped_labels


# wczytanie danych
data = load_breast_cancer()
X = data.data
y = data.target

# podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# standaryzacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. ZASTOSOWANIE WŁASNEGO SimpleGMM
mygmm = SimpleGMM(n_components=2, max_iter=50, random_state=42)
mygmm.fit(X_train_scaled)

gmm_cluster_labels = mygmm.predict(X_test_scaled)
gmm_pred_labels = cluster_to_true(y_test, gmm_cluster_labels)

accuracy_gmm = accuracy_score(y_test, gmm_pred_labels)
print(f"Dokładność Simple GMM: {accuracy_gmm * 100:.6f}%")

conf_matrix = confusion_matrix(y_test, gmm_pred_labels)

plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, cmap='viridis')
plt.title("Macierz pomyłek - Prosty GMM")
plt.colorbar()
plt.xlabel("Przewidywane")
plt.ylabel("Prawdziwe")
plt.xticks([0, 1], data.target_names)
plt.yticks([0, 1], data.target_names)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, str(conf_matrix[i, j]),
                 ha='center', va='center',
                 color='white',
                 fontsize=12,
                 fontweight='bold')

plt.tight_layout()
plt.savefig("confusion_matrix_simple_gmm.png")

# 4. IMPLEMENTACJA GMM ZE SCIKIT-LEARN
sk_gmm = GaussianMixture(
    n_components=2,
    covariance_type='full',
    random_state=42
)
sk_gmm.fit(X_train_scaled)

sk_clusters = sk_gmm.predict(X_test_scaled)
sk_labels = cluster_to_true(y_test, sk_clusters)

# 5. ZASTOSOWANIE GMM ZE SCIKIT-LEARN
accuracy_sk_gmm = accuracy_score(y_test, sk_labels)
print(f"Dokładność GMM ze sklearn: {accuracy_sk_gmm * 100:.6f}%")

conf_matrix_sk = confusion_matrix(y_test, sk_labels)

plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix_sk, cmap='plasma')
plt.title("Macierz pomyłek - GMM (sklearn)")
plt.colorbar()
plt.xlabel("Przewidywane")
plt.ylabel("Prawdziwe")
plt.xticks([0, 1], data.target_names)
plt.yticks([0, 1], data.target_names)

for i in range(conf_matrix_sk.shape[0]):
    for j in range(conf_matrix_sk.shape[1]):
        plt.text(j, i, str(conf_matrix_sk[i, j]),
                 ha='center', va='center',
                 color='white',
                 fontsize=12,
                 fontweight='bold')

plt.tight_layout()
plt.savefig("confusion_matrix_sklearn_gmm.png")
