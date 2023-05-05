import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import data as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns

from scipy import stats

import numpy as np

def slipXandY(data):
    X = data.copy()
    X = X.drop(['TARGET','DAY_ID'], axis=1)
    print("X columns 1: ", X.columns)
    y = data['TARGET'].values
    return X,y


def SimpleLinearRegression(X,y):
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    print("R2 score:", r2_score(y, y_pred))
    plt.plot(X,y)
    plt.show()
    return model

def create_regression_data(dataX, dataY):

    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=None)
    return x_train, x_test, y_train, y_test


scaler = StandardScaler()


def prepare_data(data):
    data = data.drop(['ID', 'COUNTRY', 'DAY_ID'], axis=1)
    names = data.columns
    scaled_data = scaler.fit_transform(data)
    usable_data = pd.DataFrame(scaled_data, columns=names)
    y_label = ["TARGET"]
    X = usable_data[names]
    del X["TARGET"]
    y = data[y_label]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    return x_train, x_test, y_train, y_test

def simple_regression(x_train, x_test, y_train, y_test):
    model_1_regression = LinearRegression()
    model_1_regression.fit(x_train, y_train)
    y_test_prediction = model_1_regression.predict(x_test)
    r2 = r2_score(y_test, y_test_prediction)
    spearman_corr = stats.spearmanr(y_test, y_test_prediction)[0]
    mse = mean_squared_error(y_test, y_test_prediction)
    print(f"Score / training data: {round(model_1_regression.score(x_train, y_train) * 100, 1)} %")
    print(f"Score / test data: {round(model_1_regression.score(x_test, y_test) * 100, 1)} %")
    print("R2 score : ", r2)
    print("spearman_corr : ", spearman_corr)
    print("mean squared error : ", mse)
    #print("y_test_prediction = \n", y_test_prediction)
    return model_1_regression

def test_score(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    spearman_corr = stats.spearmanr(y_test, y_pred)[0]
    mse = mean_squared_error(y_test, y_pred)

    return r2, spearman_corr, mse

def display_feat_imp_reg(reg):
    feat_imp_reg = reg.coef_[0]
    reg_feat_importance = pd.DataFrame(columns=["Feature Name", "Feature Importance"])
    reg_feat_importance["Feature Name"] = pd.Series(reg.feature_names_in_)
    reg_feat_importance["Feature Importance"] = pd.Series(feat_imp_reg)
    reg_feat_importance.plot.barh(y="Feature Importance", x="Feature Name", title="Feature importance", color="red")
    plt.show()

