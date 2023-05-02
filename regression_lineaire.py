import numpy as np
import pandas as pd
import data as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

class SimpleRegression:
    def __init__(self,data):
        self.usabled_data = data

    def slipXandY(self):
        X = self.usabled_data.copy()
        X = X.drop(['TARGET','DAY_ID'], axis=1)
        print("X columns 1: ", X.columns)
        y = self.usabled_data['TARGET'].values
        return X,y


    def SimpleLinearRegression(self,X,y):
        model = LinearRegression()
        model.fit(X,y)
        y_pred = model.predict(X)
        print("R2 score:", r2_score(y, y_pred))
        return model







