from data import *
import numpy as np
import data_analyse as da

# Press the green button in the gutter to run the script.
X, Y, combined_data = da.dt.create_data()

features = ['DE_CONSUMPTION', 'FR_CONSUMPTION', 'DE_FR_EXCHANGE',
            'FR_DE_EXCHANGE', 'DE_NET_EXPORT', 'FR_NET_EXPORT', 'DE_NET_IMPORT',
            'FR_NET_IMPORT', 'DE_GAS', 'FR_GAS', 'DE_COAL', 'FR_COAL', 'DE_HYDRO',
            'FR_HYDRO', 'DE_NUCLEAR', 'FR_NUCLEAR', 'DE_SOLAR', 'FR_SOLAR',
            'DE_WINDPOW', 'FR_WINDPOW', 'DE_LIGNITE', 'DE_RESIDUAL_LOAD',
            'FR_RESIDUAL_LOAD', 'DE_RAIN', 'FR_RAIN', 'DE_WIND', 'FR_WIND',
            'DE_TEMP', 'FR_TEMP', 'GAS_RET', 'COAL_RET', 'CARBON_RET']

print("Setting combined data")


#Remplir les données
usable_data = replace_missing_values(combined_data)
print("data : ", usable_data)
# données standardisées :

standardized_data, Y_data = create_standardized_data(usable_data)

print(da.showACP(standardized_data, Y_data))
da.showACP(standardized_data, Y_data)
standardized_data_EDA = pd.concat([standardized_data, Y], axis = 1)

da.EDA(standardized_data_EDA)

print("spliting data by country")
combined_data_used_FR = da.dt.separating_data(combined_data, 'FR')
combined_data_used_DE = da.dt.separating_data(combined_data, 'DE')
