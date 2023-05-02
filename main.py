from data import *
import numpy as np
import data_analyse as da
import regression_lineaire as rl
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# Press the green button in the gutter to run the script.
class Main :

    def __init__(self):
        self.dataFR = Data('FR')
        self.dataDE = Data('DE')
        self.dataFR = self.dataFR.create_usable_data()
        self.dataDE = self.dataDE.create_usable_data()
        self.dataNewFR = Data('FR')
        self.dataNewDE = Data('DE')
        self.dataNewFR = self.dataNewFR.create_usable_data()
        self.dataNewDE = self.dataNewDE.create_usable_data()

    def main(self):
        dataTools = da.AnalysData(self.dataFR)
        dataTools.EDA()
        SimpleRegression = rl.SimpleRegression(self.dataFR)
        X,y = SimpleRegression.slipXandY()
        self.y = y
        influenced_by =["FR_WINDPOW","DE_WINDPOW","DE_NET_IMPORT","DE_RESIDUAL_LOAD","DE_HYDRO"]
        selected_cols = X.loc[:, influenced_by]
        CPAofX = da.AnalysData(selected_cols)
        X = CPAofX.PCA_perf(1)
        SimpleRegFR = SimpleRegression.SimpleLinearRegression(X,y)

        dataTools = da.AnalysData(self.dataDE)
        dataTools.EDA()
        SimpleRegression = rl.SimpleRegression(self.dataDE)
        X,y = SimpleRegression.slipXandY()
        self.y = y
        influenced_by =["DE_RESIDUAL_LOAD","DE_NET_IMPORT","DE_GAS","DE_NET_EXPORT","DE_WINDPOW"]
        selected_cols = X.loc[:, influenced_by]
        CPAofX = da.AnalysData(selected_cols)
        X = CPAofX.PCA_perf(1)

        simpleRegDE = SimpleRegression.SimpleLinearRegression(X,y)
        self.evaluate(SimpleRegFR)




        return

    def evaluate(self,model):
        y_pred = model.predict(self.dataNewFR)
        spearman_corr, _ = spearmanr(self.y, y_pred)
        rmse = mean_squared_error(self.y, y_pred, squared=False)
        print("Spearman correlation coefficient:", spearman_corr)
        print("RMSE:", rmse)
        return spearman_corr, rmse


Main().main()
