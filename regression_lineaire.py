import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import data as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    if model == RandomForestRegressor():
        y_train = np.ravel(y_train)
    model.fit(x_train, y_train)
    y_test_prediction = model.predict(x_test)

    r2 = r2_score(y_test, y_test_prediction)
    spearman_corr = stats.spearmanr(y_test, y_test_prediction)[0]
    mse = mean_squared_error(y_test, y_test_prediction)
    print(f"Score / training data: {round(model.score(x_train, y_train) * 100, 1)} %")
    print(f"Score / test data: {round(model.score(x_test, y_test) * 100, 1)} %")

    #print("y_test_prediction = \n", y_test_prediction)
    return r2, spearman_corr, mse



def display_feat_imp_reg(reg):
    feat_imp_reg = reg.coef_[0]
    reg_feat_importance = pd.DataFrame(columns=["Feature Name", "Feature Importance"])
    reg_feat_importance["Feature Name"] = pd.Series(reg.feature_names_in_)
    reg_feat_importance["Feature Importance"] = pd.Series(feat_imp_reg)
    reg_feat_importance.plot.barh(y="Feature Importance", x="Feature Name", title="Feature importance", color="red")
    plt.show()

