
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


DEBUG = False
REMOVE_OUTLIERS = False
TARGET = "Churn"


def determine_outlier_thresholds_iqr(df, col_name, th1=0.25, th3=0.75):
    # for removing outliers using Interquartile Range or IQR
    quartile1 = df[col_name].quantile(th1)
    quartile3 = df[col_name].quantile(th3)
    iqr = quartile3 - quartile1
    upper_limit = quartile3 + 1.5 * iqr
    lower_limit = quartile1 - 1.5 * iqr
    return lower_limit, upper_limit

def print_missing_values_table(data, na_name=False):
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (data[na_columns].isnull().sum() / data.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def preprocess_df(data):
    print('\nPreprocessing data...')
    target = TARGET
    total_rows_number = data.shape[0]

    columns = data.columns.to_list()

    if DEBUG:
        print(f'Total rows: {total_rows_number}, unique {col}s: {data[col].nunique()}')
    data.drop('CustomerID', axis=1, inplace=True)

    nan_cols = data.columns[data.isnull().any()].to_list()

    if DEBUG and nan_cols:
        print(f'\nColumns with nulls:\n{nan_cols}')
        print_missing_values_table(data, na_name=True)

    # fix missing values - fill with median values
    if nan_cols:
        if DEBUG:
            print(f'\nFixing missing values...')
        data.loc[:, nan_cols] = data.loc[:, nan_cols].fillna(data.loc[:, nan_cols].median())

    if DEBUG:
        nan_cols = data.columns[data.isnull().any()].to_list()
        print(f' Any columns with nulls left? {nan_cols}')

    if REMOVE_OUTLIERS:
      # print(f"\nRemoving {col} outliers using the Standard deviation method")
      # lower, upper = determine_outlier_thresholds_sdm(data, col, 6) # 4
      # print(" upper limit:", upper)
      # print(" lower limit:", lower)
      print(f"\nRemoving Tenure outliers using IQR")
      lower, upper = determine_outlier_thresholds_iqr(data, 'Tenure', th1=0.05, th3=0.95)
      print(" upper limit:", upper)
      print(" lower limit:", lower)
      data = data[(data['Tenure'] >= lower) & (data['Tenure'] <= upper)]

    print(
        f'\nFinal number of records: {data.shape[0]} / {total_rows_number} =',
        f'{data.shape[0]/total_rows_number*100:05.2f}%\n',
    )
    return data

def preprocess_data(df, ord_enc, fit_enc=False):
    # fix missing values, remove outliers
    df = preprocess_df(df)

    # encode categorical
    categorical_features = df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_features):
        if DEBUG:
            print('OrdinalEncoder categorical_features:', list(categorical_features))
        # import ordinal encoder from sklearn
        # ord_enc = OrdinalEncoder()
        if fit_enc:
            # Fit and Transform the data
            df[categorical_features] = ord_enc.fit_transform(df[categorical_features])
            df.to_pickle('clean_data.pkl')
            if DEBUG:
                print(' OrdinalEncoder categories:', ord_enc.categories_)
        else:
            # Only Transform the data (using pretrained encoder)
            df[categorical_features] = ord_enc.transform(df[categorical_features])

    columns = df.columns.to_list()
    if DEBUG and TARGET in columns and len(df) > 10:
        corr = df.corr(numeric_only=True)[TARGET]
        print(f'\nCorrelation-2 to {TARGET}:\n{corr.to_string()}')

    return df

    
if __name__ == '__main__':

  df = pd.read_csv('E Commerce Dataset.xlsx.csv')

  ord_enc = OrdinalEncoder()
  df = preprocess_data(df, ord_enc, fit_enc=True)
