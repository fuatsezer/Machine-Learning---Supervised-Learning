import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

model_adları = ["LogisticRegression","GaussianNB","KNeighborsClassifier",
                "SVC","MLPClassifier","DecisionTreeClassifier","RandomForestClassifier",
                "GradientBoostingClassifier"]
modeller = [LogisticRegression(solver = "liblinear"),GaussianNB(),KNeighborsClassifier(),
            SVC(kernel = "rbf"),MLPClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),
            GradientBoostingClassifier()]

ad_list= []
acc_list=[]
liste=[]
for model,ad in zip(modeller,model_adları):
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    liste.append([ad,acc])
    ad_list.append(add)
    acc_list.append(acc)
#%%
import pandas as pd
df = pd.DataFrame({"ad":model_adları,"acc":acc_list})
#%%
import seaborn as sns
sns.barplot(x="acc",y="ad",data=df)    


