from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
parametersGrid = {"max_iter": [1, 5, 10],
                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}
eNet = ElasticNet()
grid = GridSearchCV(eNet, parametersGrid, scoring='accuracy', cv=10)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
