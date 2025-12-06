
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.utils.helper_functions import preprocess

def random_forest(X_train, y_train):
    rf = Pipeline([('prep', preprocess()),('model',RandomForestClassifier(n_estimators=300, max_depth=13, max_features='log2', bootstrap=False, random_state=9))])
    rf.fit(X_train, y_train)
    return rf
