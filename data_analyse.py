
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
        print(coor_matrix['TARGET'].sort_values(ascending=True))
        sns.heatmap(coor_matrix, annot=True)
        plt.show()

        # Tracer les graphes respectifs avec en ordonnées TARGET et absices les variables DE_NET_EXPORT, DE_WINDPOW, FR_WINDPOW, DE_RESIUDAL_LOAD, DE_NET_IMPORT
        print("Scatter Plot")
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        axs[0].scatter(usuabled_data['DE_NET_EXPORT'], usuabled_data['TARGET'])
        axs[1].scatter(usuabled_data['DE_WINDPOW'], usuabled_data['TARGET'])
        axs[2].scatter(usuabled_data['FR_WINDPOW'], usuabled_data['TARGET'])
        axs[3].scatter(usuabled_data['DE_RESIDUAL_LOAD'], usuabled_data['TARGET'])
        axs[4].scatter(usuabled_data['DE_NET_IMPORT'], usuabled_data['TARGET'])
        axs[0].set_xlabel('DE_NET_EXPORT')
        axs[0].set_ylabel('TARGET')
        axs[1].set_xlabel('DE_WINDPOW')
        axs[1].set_ylabel('TARGET')
        axs[2].set_xlabel('FR_WINDPOW')
        axs[2].set_ylabel('TARGET')
        axs[3].set_xlabel('DE_RESIDUAL_LOAD')
        axs[3].set_ylabel('TARGET')
        axs[4].set_xlabel('DE_NET_IMPORT')
        axs[4].set_ylabel('TARGET')
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

    def PCA(self):
        pca = PCA(n_components=2)
        pca.fit(self.data.create_usable_data())
        pca_variance_ratio = pca.explained_variance_ratio_
        pca_covariance = pca.get_covariance()
        print("PCA")
        print("PCA columns name" , pca.get_params())
        print("PCA components:", pca.components_)
        print("PCA explained variance ratio:", pca.explained_variance_ratio_)
        return pca, pca_variance_ratio, pca_covariance