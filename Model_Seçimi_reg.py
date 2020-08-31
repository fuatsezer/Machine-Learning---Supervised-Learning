from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
model_adları = ["Lineer_Regresyon","ElasticNetCV","ElasticNet","Lasso","Ridge","PLSRegression",
                "KNeighborsRegressor","SVR","MLPRegressor","BaggingRegressor",
                "RandomForestRegressor","GradientBoostingRegressor",
                "CatBoostRegressor"]
modeller = [LinearRegression(),ElasticNetCV(cv = 10, random_state = 0),
            ElasticNet(),Lasso(alpha = 0.1),Ridge(alpha = 0.1),
            PLSRegression(),KNeighborsRegressor(),SVR(kernel="rbf"),
            MLPRegressor(hidden_layer_sizes = (100,20)),
            BaggingRegressor(bootstrap_features = True),
            RandomForestRegressor(random_state = 42),GradientBoostingRegressor(),
            CatBoostRegressor()]
ad_list= []
rmse_list=[]
liste=[]
for model,ad in zip(modeller,model_adları):
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    liste.append([ad,rmse])
    ad_list.append(add)
    rmse_list.append(rmse)
#%%
import pandas as pd
df = pd.DataFrame({"ad":model_adları,"rmse":rmse_list})
#%%
import seaborn as sns
sns.barplot(x="rmse",y="ad",data=df)    





