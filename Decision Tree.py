import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sklearn as sklearn
 
dataset = pd.read_csv('OG_Dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(splitter = "random", max_depth=4, min_samples_split = 4, min_samples_leaf= 4, max_features = 2)
dt.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = dt.predict(x_test)
print(y_pred)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")
acc=r2*100
print("R-squared times 100:",acc)
pred_df=pd.DataFrame({ 'Actual Value':y_test,'Predicted Value':y_pred, 'Difference': y_test-y_pred})
pred_df.to_csv('ActualvsPredict(DecisionTree).csv')
y_pred=y_pred.reshape(len(y_pred),1)
TempnpH = pd.read_csv('new_file.csv')
Temp = TempnpH.iloc[:, :-1].values
pH = TempnpH.iloc[:, -1].values
y_pred= y_pred.reshape(len(y_pred),1)
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Temp, pH, y_pred, cmap='coolwarm')
ax.set_xlabel('Temperature')
ax.set_ylabel('pH')
ax.set_zlabel('Predicted Concentration')
plt.savefig('DT.png')

import matplotlib.pyplot as plt
import pandas as pd
 
pred_df=pd.read_csv('ActualvsPredict(DecisionTree).csv')
 
fig, ax = plt.subplots()
ax.scatter(pred_df['Actual Value'], pred_df['Predicted Value'], c='blue')
ax.plot(pred_df['Actual Value'], pred_df['Actual Value'], c='red')
ax.set_xlabel('Actual Value')
ax.set_ylabel('Predicted Value')
ax.set_title('Actual vs Predicted Values (Decision Tree Regression)')

plt.savefig('Actual_vs_Predicted_DT.png')


from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
dot_data = export_graphviz(dt, out_file=None, 
                           feature_names=['Temperature', 'pH'],  
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = graphviz.Source(dot_data)  

graph.render(filename='decision_tree', format='pdf')
