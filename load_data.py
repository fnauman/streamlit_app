import pandas as pd
from sklearn import datasets


def load_data():
    data = datasets.load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    return df

