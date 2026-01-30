import great_expectations as gx
import pandas as pd
from pathlib import Path

from great_expectations.dataset import PandasDataset

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "scaled_data_clusters.csv"

def validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected data file at: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    ge_df = PandasDataset(df)

    required_cols = [
        'Glucose', 'Age', 'BloodPressure', 'SkinThickness',
        'BMI', 'Insulin_log', 'DiabetesPedigreeFunction_log', 'Cluster'
    ]
    for col in required_cols:
        ge_df.expect_column_to_exist(col)


    for col in ['Glucose', 'Age', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin_log', 'DiabetesPedigreeFunction_log']:
        ge_df.expect_column_values_to_not_be_null(col)

    ge_df.expect_column_values_to_be_between('Age', min_value=18, max_value=100)
    ge_df.expect_column_values_to_be_in_set('Cluster', {0, 1})

    result = ge_df.validate()
    if not result['success']:
        raise ValueError('data quality failed')
    else:
        print('data success')

if __name__ == "__main__":
    validate_data()