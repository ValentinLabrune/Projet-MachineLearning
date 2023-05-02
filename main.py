from data import *
import numpy as np
import data_analyse as da
import regression_lineaire as rl

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    new_data = Data()
    new_data.CombiningDataFrame()
    new_rl = ["DE_NET_EXPORT", "DE_WINDPOW", "FR_WINDPOW", "DE_RESIDUAL_LOAD", "DE_NET_IMPORT"]
    mul = rl.SimpleRegression(new_rl)
    LinearRegression.SimpleLinearRegression()


