from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

podaci = pd.read_csv(r'C:\Users\dudic\Desktop\SetA.csv')
podaci = podaci.replace(np.nan, 0)
X=podaci[['airTemp','waterTemp', 'salinity', 'pH', 'turbidity', 'sunRadiation', 'rainfall', 'rainfall24', 'rainfall48', 'rainfall72', 'windDirection', 'windStrength', 'numBathers' ]].values
y=podaci['EC'].values

features = podaci.columns

#random forest
model1 = RandomForestClassifier(n_estimators = 50, n_jobs=2, random_state = 1)
model1.fit(X,y)

#linearna regresija
model2 = LinearRegression()
model2.fit(X,y)

importances1 = model1.feature_importances_
indices1 = np.argsort(importances1)


#utjecajnost parametara prikazana grafom - linearna regresija
importance = model2.coef_
for i,v in enumerate(importance):
	print("Feature: ", podaci.columns[i], " ,Score: ", v)
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


#utjecajnost parametara prikazana grafom - random forest
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices1)), importances1[indices1], color='red', align='center')
plt.yticks(range(len(indices1)), features[indices1])
plt.xlabel('Relative Importance')
plt.show()

#korelacija parametara
corr = podaci[['airTemp','waterTemp', 'salinity', 'pH', 'turbidity', 'sunRadiation', 'rainfall', 'rainfall24', 'rainfall48', 'rainfall72', 'windDirection', 'windStrength', 'numBathers' ]].corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);