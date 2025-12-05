import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    """Train and return a 5-NN classifier."""
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    
    model.fit(X_train, y_train) 
    return model

