import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sklearn

dataset = pd.read_csv('OG_Dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.svm import SVR
regressor = SVR(kernel = 'linear', gamma= 'scale')
regressor.fit(x,y)
A=regressor.predict(x)
pred_df=pd.DataFrame({ 'Actual Value':y,'Predicted Value':A, 'Difference': y-A})
pred_df.to_csv('ActualvsPredict(SVR)-linear.csv')


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y, A)
print(f"Mean squared error: {mse:.2f}")
r2 = r2_score(y, A)
print(f"R-squared: {r2:.2f}")
acc=r2*100
print("R-squared times 100:",acc)
y=y.reshape(len(y),1)
plt.figure(figsize=(15,7))
# Errors
ax_x= pred_df['Actual Value']
ax_y= pred_df['Predicted Value']
yerr= pred_df['Difference']

plt.scatter(range(len(yerr)), yerr, alpha=.5)
plt.title('Homogeneous Errors - SVR', size=15);
plt.hlines(y=0, xmin=0, xmax=11000, linestyle='--', color='white', alpha=.5);
plt.ylim(-.3, .3)
plt.savefig('SVR_Error-linear.png')

Temp = dataset.iloc[:, :-2].values
pH= dataset.iloc[:, -2].values
A=A.reshape(len(A),1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Temp, pH, A, cmap='coolwarm')
ax.set_xlabel('Temperature')
ax.set_ylabel('pH')
ax.set_zlabel('Predicted Concentration')
plt.savefig('SVR-linear.png')




