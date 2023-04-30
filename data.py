import pandas as pd
from sklearn.decomposition import PCA

class Data:
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
        # print the number of columns
        print(self.combined_data)

    def printIndexOfNullData(self):
        missing = pd.isnull(pd.isnull(self.combined_data))
        print("ok")
        missing[missing.isin(['true'])].stack()
        print("---")

    def replaceMissingValues(self, threshold=0.1):
        # Replace missing values with the mean of the column if the percentage of missing values is less than threshold
        num_rows = self.combined_data.shape[0]
        for col in self.combined_data.columns:
            num_missing = self.combined_data[col].isnull().sum()
            if num_missing > 0:
                if num_missing / num_rows <= threshold:
                    mean = self.combined_data[col].mean()
                    self.combined_data[col].fillna(mean, inplace=True)
                else:
                    self.combined_data.dropna(subset=[col], inplace=True)

    def create_usable_data(self):
        self.replaceMissingValues()
        usable_data = self.combined_data.drop(['ID', 'COUNTRY'], axis=1)
        print("abc\n", usable_data)
        return usable_data

    def perform_pca(self, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(self.create_usable_data())
        pca_variance_ratio = pca.explained_variance_ratio_
        pca_covariance = pca.get_covariance()
        return pca, pca_variance_ratio, pca_covariance
