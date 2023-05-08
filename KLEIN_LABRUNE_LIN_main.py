from matplotlib import pyplot as plt

from KLEIN_LABRUNE_LIN_data import *
import numpy as np
import KLEIN_LABRUNE_LIN_data_analyse as da
import KLEIN_LABRUNE_LIN_model as rl
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Press the green button in the gutter to run the script.
X, Y, combined_data, data_new = da.dt.create_data()
#features = ['DE_CONSUMPTION', 'FR_CONSUMPTION', 'DE_FR_EXCHANGE','FR_DE_EXCHANGE', 'DE_NET_EXPORT', 'FR_NET_EXPORT', 'DE_NET_IMPORT','FR_NET_IMPORT', 'DE_GAS', 'FR_GAS', 'DE_COAL', 'FR_COAL', 'DE_HYDRO','FR_HYDRO', 'DE_NUCLEAR', 'FR_NUCLEAR', 'DE_SOLAR', 'FR_SOLAR','DE_WINDPOW', 'FR_WINDPOW', 'DE_LIGNITE', 'DE_RESIDUAL_LOAD','FR_RESIDUAL_LOAD', 'DE_RAIN', 'FR_RAIN', 'DE_WIND', 'FR_WIND','DE_TEMP', 'FR_TEMP', 'GAS_RET', 'COAL_RET', 'CARBON_RET']

print("Setting combined data")


#Remplir les données
usable_data = replace_missing_values(combined_data)
usable_data = suppressing_absurd_data(usable_data)
print("Usable data EDA")
da.EDA(usable_data)
# données standardisées :
standardized_data, Y_data = create_standardized_data(usable_data)
#print("données finales = ", standardized_data, Y_data)
#print(da.showACP(standardized_data, Y_data))




print("spliting data by country")
combined_data_used_FR = da.dt.separating_data(combined_data, 'FR')
combined_data_used_DE = da.dt.separating_data(combined_data, 'DE')

#Sert à voir quelles données seront à virer
print(X.dtypes)

print(standardized_data.shape, Y_data.shape)

x_train1, x_test1, y_train1, y_test1 = rl.create_regression_data(standardized_data, Y_data)

print("ShapeXtrain1", x_train1.shape, "ShapeXtest", x_test1.shape, "ShapeYtrain1", y_train1.shape, "shapeytest", y_test1.shape)
#print("Xtrain = ", x_train1, "Xtest",x_test1, "ytrain", y_train1, "ytest", y_test1)

#evaluate model
models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge(alpha=0.5)),
    ('Lasso Regression', Lasso(alpha = 0.02)),
    ('K-Nearest Neighbors', KNeighborsRegressor()),
    ('Decision Tree', DecisionTreeRegressor(max_depth=5)),
    ('Random Forest', RandomForestRegressor(max_depth= 5))
]

accuracy = []
for name, model in models:
    print(f"{name}:")
    r2, spearman_corr, mse = rl.train_and_evaluate_by_model(model, x_train1, x_test1, y_train1, y_test1)
    accuracy.append((name, r2, mse, spearman_corr))
    print(f"  R2: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  Spearman Correlation: {spearman_corr:.4f}\n")
    if name != "K-Nearest Neighbors" and name != "Decision Tree" and name != "Random Forest":
        rl.display_feat_imp_reg(model)

# Comparez les performances des modèles
results_df = pd.DataFrame(accuracy, columns=['Model', 'R2', 'MSE', 'Spearman Correlation'])
# Score is egal to (r2 - mse + spearman_corr) / 3
results_df['Score'] = (results_df['R2'] - results_df['MSE'] + results_df['Spearman Correlation']) / 3
results_df = results_df.sort_values(by='Score', ascending=False)
print("Performance ranking:")
print(results_df)


fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
x = np.arange(len(results_df))

ax.bar(x - bar_width, results_df['MSE'], width=bar_width, label='MSE')
ax.bar(x, results_df['R2'], width=bar_width, label='R2')
ax.bar(x + bar_width, results_df['Spearman Correlation'], width=bar_width, label='Spearman Correlation')

ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=45)
ax.legend()

plt.title("Performance Comparison")
plt.show()

#Récupérer les ids
data_new_ID = data_new['ID']
usable_new_data = replace_missing_values(data_new)
#Usable new_data only keeps the same columns as usable_data
usable_new_data = usable_new_data[standardized_data.columns]
#Standardize the new data
standardized_new_data = create_standardized_data_new(usable_new_data)
#Predict the values
y_pred = rl.predict_data(standardized_new_data, x_train1, y_train1, data_new_ID)



#print(y_test_prediction)


# UAL_LOAD, FR_CONSUMP, DE_NUCLEAR