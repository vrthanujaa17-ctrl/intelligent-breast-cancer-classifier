from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def variance_filter(X_train, X_test, threshold: float = 0.01):
    """
    Remove low-variance features.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_var = selector.fit_transform(X_train)
    X_test_var = selector.transform(X_test)
    return X_train_var, X_test_var, selector


def apply_pca(X_train, X_test, n_components=0.95):
    """
    Apply PCA while preserving specified variance.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca