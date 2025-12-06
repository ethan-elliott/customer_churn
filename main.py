from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.models.decision_tree import train_decision_tree
from src.models.train_model import split_data, plot_roc_curve
from src.models.knn_model import train_knn_model
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)
from src.models.forest import random_forest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main() -> None:
    print("---Loading data...")
    raw_df = load_dataset("data/raw/train.csv") #Labelled data for supervised learning
    raw_test = load_dataset("data/raw/test.csv") #Unlabeled test data

    print("---Cleaning data...")
    clean_df = clean_dataset(raw_df)
    clean_df.to_csv('./data/processed/clean_data')
    clean_test = clean_dataset(raw_test)
    clean_test.to_csv('./data/processed/clean_test') #Unlabeled test data
    X_cols_knn = ['Usage Frequency', 'Support Calls','Age']
    X_cols_tree = ['Age','Tenure','Usage Frequency','Support Calls','Total Spend','Last Interaction','Contract Length','Gender','Subscription Type']

    print("---Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean_df)
    X_test_knn = X_test[X_cols_knn]
    X_test_tree = X_test[X_cols_tree]

    print("---Training models...")
    knn_model = train_knn_model(X_train[X_cols_knn], y_train)
    tree = train_decision_tree(X_train[X_cols_tree], y_train)
    forest = random_forest(X_train, y_train) #added later because it got a better score

    print("---Evaluating on validation set...")
    y_val_pred_knn = knn_model.predict(X_val[X_cols_knn])
    y_val_pred_tree = tree.predict(X_val[X_cols_tree])
    y_val_pred_forest = forest.predict(X_val)

    val_prob_knn = knn_model.predict_proba(X_val[X_cols_knn])[:, 1]
    val_prob_tree = tree.predict_proba(X_val[X_cols_tree])[:, 1]
    val_prob_forest = forest.predict_proba(X_val)[:, 1]

    plot_confusion_matrices(y_val, y_val_pred_tree, y_val_pred_knn)
    plot_performance_comparison(y_val, y_val_pred_tree, y_val_pred_knn)

    auc_tree = plot_roc_curve(y_val, val_prob_tree, "Decision Tree")
    auc_knn = plot_roc_curve(y_val, val_prob_knn, "5-NN")
    auc_forest = plot_roc_curve(y_val, val_prob_forest, "Random Forest")

    best_model = forest
    best_label = "Random Forest"

    print(f"---Testing best model ({best_label})...")
    y_test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, test_prob, f"Test {best_label}")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print(f'---Applying best model ({best_label}) to unlabeled test data...')
    submission = pd.DataFrame(clean_test['CustomerID'])
    submission['Churn'] = best_model.predict_proba(clean_test[X_cols_tree])[:,1]
    submission.to_csv('./data/submission/best_guess.csv', index=False)
    print("Done.")


if __name__ == "__main__":
    main()
