import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sklearn as sklearn

dataset = pd.read_csv('OG_Dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=790, weights = 'uniform', algorithm = 'kd_tree')
regressor.fit(x,y)
A=regressor.predict(x)
pred_df=pd.DataFrame({ 'Actual Value':y,'Predicted Value':A, 'Difference': y-A})
pred_df.to_csv('ActualvsPredict(KNN).csv')

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y, A)
print(f"Mean squared error: {mse:.2f}")
r2 = r2_score(y, A)
print(f"R-squared: {r2:.2f}")
acc=r2*100
print("R-squared times 100:",acc)
y=y.reshape(len(y),1)
import seaborn as sns 


plt.figure(figsize=(15,7))
# Errors
ax_x= pred_df['Actual Value']
ax_y= pred_df['Predicted Value']
yerr= pred_df['Difference']

plt.scatter(range(len(yerr)), yerr, alpha=.5)
plt.title('Homogeneous Errors - KNN', size=15);
plt.hlines(y=0, xmin=0, xmax=11000, linestyle='--', color='white', alpha=.5);
plt.ylim(-.3, .3)
plt.savefig('KNN.png')

plt.scatter(ax_x, ax_y, alpha=.5)
plt.title('KNN Scatter Plot', size=15)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.hlines(y=np.mean(y), xmin=np.min(ax_x), xmax=np.max(ax_x), linestyle='--', color='red', alpha=.5, label='Mean')
plt.legend()
plt.savefig('KNN_ScatterPlot.png')


Temp = dataset.iloc[:, :-2].values
pH= dataset.iloc[:, -2].values
A=A.reshape(len(A),1)
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Temp, pH, A, cmap='coolwarm')
ax.set_xlabel('Temperature')
ax.set_ylabel('pH')
ax.set_zlabel('Predicted Concentration')
plt.savefig('KNN_3D.png')

from scipy.stats import probplot
probplot(yerr, dist='norm', plot=plt)
plt.savefig('KNN_ProbPlot.png')

     

