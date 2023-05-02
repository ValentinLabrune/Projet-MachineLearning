import numpy as np
import pandas as pd
import data as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns

#Semble etre expliqué par les variables DE_NET_EXPORT, DE_WINDPOW, FR_WINDPOW, DE_RESIUDAL_LOAD, DE_NET_IMPORT
class SimpleRegression :
    def __init__(self,explained_variable):
        data = dt.Data()
        self.data = data.CombiningDataFrame()
        self.usabled_data = data.create_usable_data()
        self.explained_variable = explained_variable
        print(self.explained_variable)

    def SimpleLinearRegression(self):
        print(self.usabled_data.columns)
        print(self.explained_variable)
        data = self.usabled_data.loc[:, self.explained_variable + ['TARGET']]
        X = data[self.explained_variable].values
        y = data['TARGET'].values.reshape(-1, 1)
        regressor = LinearRegression()
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        plt.scatter(X.ravel(), y.ravel())
        plt.plot(X.flatten(), y_pred, color='red')
        plt.show()
        return regressor.coef_, regressor.intercept_







