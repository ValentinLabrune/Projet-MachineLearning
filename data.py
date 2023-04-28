import pandas as pd
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score

class Data :
    def __init__(self):
        self.dataX = pd.read_csv('./Données/Data_X.csv')
        self.dataY = pd.read_csv('./Données/Data_Y.csv')
        self.mergeData()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    def mergeData(self):
        print("INITING DATAFRAME")
        self.dataXuse = pd.DataFrame(self.dataX)
        self.dataYuse = pd.DataFrame(self.dataY)

    def CombiningDataFrame(self):
        print("SETTING COMBINED DATA")
        self.combined_data = pd.merge(self.dataXuse, self.dataYuse)
        print(self.combined_data)

    def printIndexOfNullData(self):
        missing = pd.isnull(pd.isnull(self.combined_data))
        print("ok")
        missing[missing.isin(['true'])].stack()
        print("---")

    def create_usable_data(self):
            usable_data = self.combined_data.dropna()
            print("abc\n", usable_data)

