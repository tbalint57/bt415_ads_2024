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