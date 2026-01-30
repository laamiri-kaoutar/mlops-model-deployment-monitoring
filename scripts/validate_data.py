import great_expectations as gx
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "scaled_data_clusters.csv"
def validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected data file at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.expect_column_value_to_not_be_null('Insulin')
    df.expect_column_value_to_not_be_null('BMI')
    df.expect_column_value_to_not_be_null('DiabetesPedigreeFunction')
    df.expect_column_value_to_not_be_null('Insulin_log')
    df.expect_column_value_to_not_be_null('Pregnancies')
    df.expect_column_value_to_not_be_null('DiabetesPedigreeFunction_log')
    df.expect_column_values_to_be_between("age", min_value=18, max_value=100)
    df.expect_column_to_exist('DiabetesPedigreeFunction_log')
    result = df.validate()
    if not result['success']:
        raise ValueError('data quality failed')
    else:
        print('data success')

if __name__ == "__main__":
    validate_data()
