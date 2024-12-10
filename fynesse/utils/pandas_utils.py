import pandas as pd


def load_csv(file_name, columns=None, column_names=None, index=None):
    df = pd.read_csv(file_name)

    if columns is None:
        return df

    df = df[columns]

    if column_names is not None:
        df.columns = column_names

    if index is None:
        return df

    df.set_index(index)
    return df


def normalise_data_frame(df, columns_to_leave_out=[]):
    """
    Normalise data frame
    """
    normalised_df = df.copy()

    feature_columns = df.columns.difference(columns_to_leave_out)
    numerical_df = df[feature_columns]

    normalised_features = (numerical_df - numerical_df.min()) / (numerical_df.max() - numerical_df.min())
    
    normalised_df[numerical_df.columns] = normalised_features

    return normalised_df