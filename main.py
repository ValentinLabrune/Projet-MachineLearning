from data import *
import numpy as np
import data_analyse as da
import regression_lineaire as rl
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Press the green button in the gutter to run the script.
X, Y, combined_data = da.dt.create_data()
#features = ['DE_CONSUMPTION', 'FR_CONSUMPTION', 'DE_FR_EXCHANGE','FR_DE_EXCHANGE', 'DE_NET_EXPORT', 'FR_NET_EXPORT', 'DE_NET_IMPORT','FR_NET_IMPORT', 'DE_GAS', 'FR_GAS', 'DE_COAL', 'FR_COAL', 'DE_HYDRO','FR_HYDRO', 'DE_NUCLEAR', 'FR_NUCLEAR', 'DE_SOLAR', 'FR_SOLAR','DE_WINDPOW', 'FR_WINDPOW', 'DE_LIGNITE', 'DE_RESIDUAL_LOAD','FR_RESIDUAL_LOAD', 'DE_RAIN', 'FR_RAIN', 'DE_WIND', 'FR_WIND','DE_TEMP', 'FR_TEMP', 'GAS_RET', 'COAL_RET', 'CARBON_RET']

print("Setting combined data")


#Remplir les données
usable_data = replace_missing_values(combined_data)
usable_data = suppressing_absurd_data(usable_data)
print("abce")
print("data : azezaezae ", usable_data)
# données standardisées :
standardized_data, Y_data = create_standardized_data(usable_data)
#print("données finales = ", standardized_data, Y_data)
#print(da.showACP(standardized_data, Y_data))
da.showACP(standardized_data, Y_data)
standardized_data_EDA = pd.concat([standardized_data, Y], axis = 1)

da.EDA(standardized_data_EDA)

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
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso(alpha = 0.4)),
    ('K-Nearest Neighbors', KNeighborsRegressor()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor())
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
results_df = results_df.sort_values(by='R2', ascending=False)
print("Performance ranking:")
print(results_df)

#print(y_test_prediction)


# UAL_LOAD, FR_CONSUMP, DE_NUCLEAR
