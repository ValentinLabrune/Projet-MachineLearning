import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import KLEIN_LABRUNE_LIN_data as dt
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import numpy as np

def slipXandY(data):
    X = data.copy()
    X = X.drop(['TARGET','DAY_ID'], axis=1)
    print("X columns 1: ", X.columns)
    y = data['TARGET'].values
    return X,y



def create_regression_data(dataX, dataY):

    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=1)
    return x_train, x_test, y_train, y_test

def train_and_evaluate_by_model(model, x_train, x_test, y_train, y_test):

    #Pca de x_train à l'aide

    model.fit(x_train, y_train)
    y_test_prediction = model.predict(x_test)
    mse = mean_squared_error(y_test, y_test_prediction)
    r2 = r2_score(y_test, y_test_prediction)
    spearman_corr = spearmanr(y_test, y_test_prediction)[0]
    print(f"  Score / training data: {round(model.score(x_train, y_train) * 100, 1)} %")
    print(f"  Score / test data: {round(model.score(x_test, y_test) * 100, 1)} %")

    #print("y_test_prediction = \n", y_test_prediction)
    return r2, spearman_corr, mse

def display_feat_imp_reg(reg):
    feat_imp_reg = reg.coef_[0]
    reg_feat_importance = pd.DataFrame(columns=["Feature Name", "Feature Importance"])
    reg_feat_importance["Feature Name"] = pd.Series(reg.feature_names_in_)
    reg_feat_importance["Feature Importance"] = pd.Series(feat_imp_reg)
    reg_feat_importance.plot.barh(y="Feature Importance", x="Feature Name", title="Feature importance", color="red")
    plt.show()

def predict_data(data, x_train, y_train, ID):
    model = Ridge(alpha=0.5)
    model.fit(x_train, y_train)
    y_test_prediction = model.predict(data)
    print("y_test_prediction columns : ", y_test_prediction.shape)
    #Add the ID
    y_test_prediction = np.column_stack((ID, y_test_prediction))
    # ADD it in the csv
    np.savetxt("données/KLEIN_LABRUNE_LIN_prediction.csv", y_test_prediction, delimiter=",", fmt='%s')
    print("y_test_prediction columns : ", y_test_prediction.shape)




