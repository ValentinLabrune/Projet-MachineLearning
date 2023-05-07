import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def create_data():
    dataX = pd.read_csv('./Données/Data_X.csv')
    dataY = pd.read_csv('./Données/Data_Y.csv')
    combined_data = pd.merge(dataX, dataY, on='ID')
    return dataX, dataY, combined_data

def replace_missing_values(data):
    threshold = 0.1
    num_rows = data.shape[0]
    for col in data.columns:
        num_missing = data[col].isnull().sum()
        if num_missing > 0:
            if num_missing / num_rows <= threshold:
                mean = data[col].mean()
                data[col].fillna(mean, inplace=True)
            else:
                data.dropna(subset=[col], inplace=True)
    return data

def suppressing_absurd_data(data):
    abc = data.copy()
    Q1 = np.percentile(abc['TARGET'], 25,
                       interpolation='midpoint')

    Q3 = np.percentile(abc['TARGET'], 75,
                       interpolation='midpoint')
    IQR = Q3 - Q1
    print("Old Shape: ", abc.shape)
    # Upper bound
    upper = np.where(abc['TARGET'] >= (Q3 + 1.5 * IQR))
    # Lower bound
    lower = np.where(abc['TARGET'] <= (Q1 - 1.5 * IQR))
    abc.drop(upper[0], inplace = True)
    abc.drop(lower[0], inplace = True)
    print("New Shape: ", abc.shape)
    return abc

def create_standardized_data(data):
    data = data.sort_values(by=['DAY_ID'])
    Y_data = data.loc[:, ['TARGET']]
    Y = pd.DataFrame(Y_data)
    print("yshape", Y.shape)
    # Dropping useless data
    usable_data = data.drop(['ID', 'COUNTRY', 'DAY_ID', 'TARGET'], axis=1)
    # Standardize data
    scaled_data = StandardScaler().fit_transform(usable_data)
    scaled_data = pd.DataFrame(scaled_data, columns=usable_data.columns)
    print(scaled_data.shape)


    return scaled_data, Y

def separating_data(combined_data, country):
    if country == 'FR':
        print('FR : ')
        combined_dataFR = combined_data.loc[combined_data['COUNTRY'] == 'FR']
        return combined_dataFR
    print('DE : ')
    combined_dataDE = combined_data.loc[combined_data['COUNTRY'] == 'DE']
    return combined_dataDE

