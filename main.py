from data import *
import numpy as np
import data_analyse as da

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    new_data = Data()
    new_data.CombiningDataFrame()
    new_da = da.AnalysData()
    new_da.EDA()
