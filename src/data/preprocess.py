import os
import pandas as pd
from .load_data import load_dataset


def clean_dataset(df):
    def clean_numeric(df, column, assume):
        df[column] = df[column].replace('none', assume)
        df[column] = pd.to_numeric(df[column])
        fill = df[column].mean()
        df[column] = df[column].fillna(fill)
        df[column] = df[column].astype('int64')
    clean_numeric(df,'Support Calls',0)
    df['Contract Length'] = df['Contract Length'].astype('category')
    df['Gender'] = df['Gender'].astype('category')
    df['Subscription Type'] = df['Subscription Type'].astype('category')
    return df

if __name__ == "__main__":
    # Load the raw dataset
    raw = load_dataset("data/raw/card_transdata.csv")
    # Clean the dataset
    cleaned = clean_dataset(raw)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "data/processed/card_transdata_clean.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
