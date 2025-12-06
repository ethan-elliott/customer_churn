# Customer Churn Prediction

This repository contains a machine learning project developed to predict customer churn based on behavioral, demographic, and usage data. The project includes reproducible preprocessing pipelines, model training workflows, and an exploratory analysis notebook.

## Project layout

```
customer_churn/
├── data/
│ ├── raw/ # Original datasets 
│ ├── processed/ # Cleaned atasets
├── notebooks/ # Jupyter notebooks for EDA 
├── src/
│ ├── data/ # Data loading and preprocessing modules
│ ├── features/ # Feature engineering utilities (currently unused)
│ ├── models/ # Model training and evaluation utilities
│ ├── utils/ # Shared helper functions
├── requirements.txt # Project dependencies
└── README.md
```

`main.py` imports the modules inside `src/` and executes them to reproduce model training and results.


## Package versions

Package versions used for this project are found in requirements.txt

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
```
