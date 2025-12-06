"""Utility helpers for the project."""
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cat_cols = ['Gender', 'Contract Length', 'Subscription Type']
num_cols = ['Age','Tenure','Usage Frequency','Support Calls','Total Spend','Last Interaction']

def preprocess():
    return ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),('num', 'passthrough', num_cols)])

