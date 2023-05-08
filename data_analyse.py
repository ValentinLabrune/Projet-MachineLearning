import numpy as np
import pandas as pd
import data as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


def ACP(dataX):
    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(dataX)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1'])
    return principalDf


def showACP(dataX, dataY):
    principalDf = ACP(dataX)
    finalDf = pd.concat([principalDf, dataY], axis=1) #.sort_values(by=['principal component 1'])
    plt.title("TARGET varie en fcontion de l'ACP")
    plt.ylabel('TARGET')
    plt.xlabel('ACP')
    plt.plot(principalDf, dataY)
    plt.show()
    return finalDf

def EDA(data):
    print('----------------------------------------------------------------')
    print("EDA")
    print("Summary Statistics")
    print(data.describe())
    print("Columns: ", data.columns)
    print("Correlation Matrix")
    #Si la colonne country est de type string, FR :0, DE :
    if data['COUNTRY'].dtype == 'object':
        data['COUNTRY'] = data['COUNTRY'].replace(['FR', 'DE'], [0, 1])
        data['COUNTRY'] = data['COUNTRY'].astype(int)
    coor_matrix = data.corr()
    print(coor_matrix['TARGET'].sort_values(ascending=True))
    plt.title('Correlation Matrix')
    plt.figure(figsize=(24, 20))
    sns.heatmap(coor_matrix, annot=True)
    plt.show()

