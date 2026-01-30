import greate_expectations as gx

def validate_data():
    df = gx.read_csv('../data/scaled_data_clusters.csv')
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
