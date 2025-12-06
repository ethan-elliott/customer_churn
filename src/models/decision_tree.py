import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from src.utils.helper_functions import preprocess

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    tree = Pipeline([('prep', preprocess()),('model', DecisionTreeClassifier(max_depth=8, random_state=9))])
    tree.fit(X_train, y_train)
    return tree
