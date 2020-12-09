# Importation des packages
import pandas as pd
import numpy as np

# Chargement de la base de données
titanic = pd.read_csv("C:\\Users\\OUATTARA Abdul-Aziz\\Downloads\\train.csv")

# Préprocessing

def convert_cast(df, categorical_cols):
    ''' Cette fonction permet de convertir les colonnes des variables catégorielles en object'''

    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype("object")
    return df


def convert_cast(mapper_config=None, dataframe=None):
    """
    Convert and cast DataFrame field types.

    This function achieves the same purpose as pd.read_csv, but it is used on
    actual DataFrame obtained from SQL.
    """
    # field conversion
    for var_name, lambda_string in mapper_config['converters'].items():
        if var_name in dataframe.columns:
            func = eval(lambda_string)
            dataframe[var_name] = dataframe[var_name].apply(func)

    # field type conversion/compression
    dty = dict(dataframe.dtypes)
    for col_name, actual_type in dty.items():
        expected_type = mapper_config['columns'][col_name]['dtype']
        if actual_type != expected_type:
            dataframe[col_name] = dataframe[col_name].astype(expected_type)

    return dataframe


# odef input_missing_values(df):
    for col in df.columns:
        if (df[col].dtype is float) or (df[col].dtype is int):
            df[col] = df[col].fillna(df[col].median())
        if (df[col].dtype == object):
            df[col] = df[col].fillna(df[col].mode()[0])

    return df



def parse_model(X, use_columns):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    X = X[use_columns]
    return X, target

