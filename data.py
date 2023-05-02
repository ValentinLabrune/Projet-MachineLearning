import pandas as pd
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Data:
    def __init__(self,country , X_new = None):
        if not X_new:
            self.dataX = pd.read_csv('./Données/Data_X.csv')
            self.dataY = pd.read_csv('./Données/Data_Y.csv')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            self.country = country
        else:
            self.dataX = pd.read_csv('./Données/DataNew_X.csv')
            self.dataY = pd.read_csv('./Données/Data_Y.csv')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)


    def SeparatingData(self):
        print("SETTING COMBINED DATA")
        dataFR = pd.DataFrame(self.dataX)
        dataDE = pd.DataFrame(self.dataX)
        dataTarget = pd.DataFrame(self.dataY)

        dataFR = dataFR.sort_values(by=['ID'])
        dataDE = dataDE.sort_values(by=['ID'])

        combined_dataFR = pd.merge(dataFR, dataTarget, on='ID')
        combined_dataDE = pd.merge(dataDE, dataTarget, on='ID')

        combined_dataFR.to_csv('./Données/combined_dataFR.csv', index=False)

        # Filter data by country
        if self.country :
            combined_dataFR = combined_dataFR.loc[dataFR['COUNTRY'] == self.country]
            combined_dataDE = combined_dataDE.loc[dataDE['COUNTRY'] == self.country]

        if self.country == 'FR':
            return combined_dataFR
        elif self.country == 'DE':
            return combined_dataDE

    def replaceMissingValues(self, data, threshold=0.1):
        # Replace missing values with the mean of the column if the percentage of missing values is less than threshold
        num_rows = data.shape[0]
        for col in data.columns:
            num_missing = data[col].isnull().sum()
            if num_missing > 0:
                if num_missing / num_rows <= threshold:
                    mean = data[col].mean()
                    data[col].fillna(mean, inplace=True)
                else:
                    data.dropna(subset=[col], inplace=True)

    def create_usable_data(self):
        combined_data = self.SeparatingData()
        self.replaceMissingValues(combined_data)
        usable_data = combined_data.drop(['ID', 'COUNTRY'], axis=1)
        #Standardize data
        usable_data_without_day_id = usable_data.drop('DAY_ID', axis=1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(usable_data_without_day_id)
        scaled_data = pd.DataFrame(scaled_data, columns=usable_data_without_day_id.columns)
        scaled_data.insert(0, 'DAY_ID', usable_data['DAY_ID'])
        scaled_data = scaled_data.sort_values(by=['DAY_ID'])
        scaled_data.to_csv('./Données/scaled_data.csv', index=False)
        return scaled_data
