import numpy as np
import pandas as pd
import KLEIN_LABRUNE_LIN_data as dt
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
    data['COUNTRY'] = data['COUNTRY'].replace(['FR', 'DE'], [0, 1])
    print('----------------------------------------------------------------')
    print("EDA")
    # Faire des histogrammes
    print("Histograms : ")
    data.hist(bins=40, figsize=(22, 18))
    plt.show()
    print("Summary Statistics")
    print(data.describe())
    print("Columns: ", data.columns)
    print("Correlation Matrix")
    coor_matrix = data.corr()
    print("coor_matrix values :", abs(coor_matrix['TARGET'].sort_values(ascending=True)))
    plt.title('Correlation Matrix')
    plt.figure(figsize=(24, 20))
    sns.heatmap(coor_matrix, annot=True)
    plt.show()
