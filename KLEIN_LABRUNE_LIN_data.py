import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def create_data():
    dataX = pd.read_csv('./Données/Data_X.csv')
    dataY = pd.read_csv('./Données/Data_Y.csv')
    new_data_X = pd.read_csv('./Données/DataNew_X.csv')
    combined_data = pd.merge(dataX, dataY, on='ID')
    return dataX, dataY, combined_data, new_data_X

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
    # Dropping useless data
    usable_data = data.drop(['ID', 'DAY_ID'], axis=1)
    #Remplacer FR par 0 et DE par 1 dans la colonne COUNTRY
    usable_data['COUNTRY'] = usable_data['COUNTRY'].replace(['FR', 'DE'], [0, 1])
    coor = usable_data.corr()
    for col in usable_data.columns:
        if abs(coor[col]['TARGET']) < 0.12:
            usable_data.drop(col, axis=1, inplace=True)

    print("Usable data: ", usable_data.columns)
    usable_data = usable_data.drop(['TARGET'], axis=1)
    scaled_data = StandardScaler().fit_transform(usable_data)
    scaled_data = pd.DataFrame(scaled_data, columns=usable_data.columns)
    print(scaled_data.shape)


    return scaled_data, Y


def create_standardized_data_new(data):
    usable_data = data
    print("Usable data: ", usable_data.columns)
    scaled_data = StandardScaler().fit_transform(usable_data)
    scaled_data = pd.DataFrame(scaled_data, columns=usable_data.columns)
    print(scaled_data.shape)

    return scaled_data
def separating_data(combined_data, country):
    if country == 'FR':
        print('FR : ')
        combined_dataFR = combined_data.loc[combined_data['COUNTRY'] == 'FR']
        return combined_dataFR
    print('DE : ')
    combined_dataDE = combined_data.loc[combined_data['COUNTRY'] == 'DE']
    return combined_dataDE