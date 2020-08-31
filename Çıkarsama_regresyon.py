import statsmodels.api as sm
lm = sm.OLS(y,X)
model = lm.fit()
print(model.summary())
#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X, y)
print(model.intercept_)
print(model.coef_)
print(model.score(X,y))


 


