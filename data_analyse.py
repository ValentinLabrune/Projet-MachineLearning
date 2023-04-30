
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
    def PCA(self, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(self.data.create_usable_data())
        pca_variance_ratio = pca.explained_variance_ratio_
        pca_covariance = pca.get_covariance()
        return pca, pca_variance_ratio, pca_covariance

    def EDA(self):
        usuabled_data = self.data.create_usable_data()

        print("Summary Statistics")
        print(usuabled_data.describe())

        print("Columns: ", usuabled_data.columns)
        print("Correlation Matrix")
        coor_matrix = usuabled_data.corr()
        print(coor_matrix['TARGET'])
        sns.heatmap(coor_matrix, annot=True)
        plt.show()

        print("Boxplot")
        usuabled_data.boxplot()
        plt.show()

        print("Histogram")
        usuabled_data.hist()
        plt.show()

        print("Distribution")
        for col in usuabled_data.columns:
            sns.distplot(usuabled_data[col], kde=False)
            plt.title(col)
            plt.show()