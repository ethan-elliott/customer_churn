import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    cat_cols = ['Gender', 'Contract Length', 'Subscription Type']
    num_cols = ['Age','Tenure','Usage Frequency','Support Calls','Total Spend','Last Interaction']
    preprocess = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),('num', 'passthrough', num_cols)])
    X_cols = ['Age','Tenure','Usage Frequency','Support Calls','Total Spend','Last Interaction','Contract Length','Gender','Subscription Type']
    X_train = X_train[X_cols]
    tree = Pipeline([('prep', preprocess),('model', DecisionTreeClassifier(max_depth=8, random_state=9))])
    tree.fit(X_train, y_train)
    return tree
