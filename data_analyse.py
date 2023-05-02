
import numpy as np
import pandas as pd
import data as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

class AnalysData:

    def __init__(self):
        data = dt.Data()
        data.CombiningDataFrame()
        self.data = data

    def PCA_perf(self, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(self.data)
        transformed = pca.transform(self.data)
        data = pd.DataFrame(transformed, columns=[f'PC{i}' for i in range(1, n_components + 1)])
        print("type of data: ", type(data))
        print("uazhegfiazoehgoazoebgoubazuoefva")
        return data

    def EDA(self):

        print("EDA for FR")
        print("Summary Statistics")
        print(self.data.describe())

        print("Columns: ", self.data.columns)
        print("Correlation Matrix")
        coor_matrix = self.data.corr()
        print(coor_matrix['TARGET'].sort_values(ascending=True))
        sns.heatmap(coor_matrix, annot=True)
        plt.show()
