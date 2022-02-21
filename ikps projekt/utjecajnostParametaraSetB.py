
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

podaci = pd.read_csv(r'C:\Users\dudic\Desktop\SetBSolcast.csv')
podaci = podaci.replace(np.nan, 0)
X=podaci[['AirTemp', 'AlbedoDaily', 'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh', 'Ghi','GtiFixedTilt', 'GtiTracking', 'PrecipitableWater', 'RelativeHumidity', 'SurfacePressure', 'WindDirection10m', 'WindSpeed10m', 'Zenith' ]].values
y=podaci['ecoli'].values


features = podaci.columns

#random forest
model1 = RandomForestClassifier(n_estimators = 50, n_jobs=2, random_state = 1)
model1.fit(X,y)

#linearna regresija
model2 = LinearRegression()
model2.fit(X,y)

importances1 = model1.feature_importances_
indices1 = np.argsort(importances1)

#utjecajnost parametara prikazana grafom - random forest
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices1)), importances1[indices1], color='red', align='center')
plt.yticks(range(len(indices1)), features[indices1])
plt.xlabel('Relative Importance')
plt.show()

#korelacija parametara
corr = podaci[['AirTemp', 'AlbedoDaily', 'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh', 'Ghi','GtiFixedTilt', 'GtiTracking', 'PrecipitableWater', 'RelativeHumidity', 'SurfacePressure', 'WindDirection10m', 'WindSpeed10m', 'Zenith' ]].corr()
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