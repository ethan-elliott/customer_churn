import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    """Train and return a 5-NN classifier."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit(X_train)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    return model
