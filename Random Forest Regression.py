import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sklearn as sklearn
dataset = pd.read_csv('OG_Dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 1000, random_state = 8, criterion= 'absolute_error',max_features='sqrt')
regressor.fit(x, y)
y_pred = regressor.predict(x)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y, y_pred)
print(f"Mean squared error: {mse:.2f}")
r2 = r2_score(y, y_pred)
print(f"R-squared: {r2:.2f}")
acc=r2*100
print("R-squared times 100:",acc)
pred_df=pd.DataFrame({ 'Actual Value':y,'Predicted Value':y_pred, 'Difference': y-y_pred})
pred_df.to_csv('ActualvsPredict(Random Forest).csv')

# plot actual vs predicted values
plt.scatter(y, y_pred, alpha=.5, label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', label='Ideal predictions')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs Predicted (Random Forest)')
plt.legend()
plt.savefig('RF.png')


# plot error distribution
y_err = y - y_pred
plt.scatter(range(len(y_err)), y_err, alpha=.5, label='Errors')
plt.title('Homogeneous Errors - Random Forest', size=15)
plt.hlines(y=0, xmin=0, xmax=11000, linestyle='--', color='white', alpha=.5)
plt.ylim(-.3, .3)
plt.legend()
plt.savefig('RF_Errors.png')


# plot 3D graph
Temp = dataset.iloc[:, :-2].values
pH = dataset.iloc[:, -2].values
y = y.reshape(len(y),1)
y_pred = y_pred.reshape(len(y_pred),1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(Temp, pH, y, c='b', marker='o', alpha=0.5, label='Actual values')
ax.scatter(Temp, pH, y_pred, c='r', marker='o', alpha=0.5, label='Predicted values')
ax.set_xlabel('Temperature')
ax.set_ylabel('pH')
ax.set_zlabel('Concentration')
plt.title('Actual vs Predicted (Random Forest)')
plt.legend()
plt.savefig('RF_3D.png')


# plot probability plot of errors
from scipy.stats import probplot
probplot(y_err, dist='norm', plot=plt)
plt.title('Probability Plot of Errors - Random Forest')
plt.legend()
plt.savefig('RF_ProbPlot.png')

from sklearn.tree import plot_tree
tree = regressor.estimators_[0] # get the first decision tree from the random forest
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True)
plt.savefig('RF_Trees.png')

