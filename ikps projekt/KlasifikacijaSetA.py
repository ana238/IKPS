import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix
from sklearn import svm
from sklearn import metrics

podaci = pd.read_csv(r'C:\Users\dudic\Desktop\SetA.csv')
podaci = podaci.replace(np.nan, 0)
podaci = podaci.astype(float)
X=podaci[['airTemp','waterTemp', 'salinity', 'pH', 'turbidity', 'sunRadiation', 'rainfall', 'rainfall24', 'rainfall48', 'rainfall72', 'windDirection', 'windStrength', 'numBathers' ]].values
y=podaci['EC'].values

podaci['EC'] = (podaci['EC'] > 300).astype(int)
print(podaci.EC.head(30))

#sns.distplot(podaci.ecoli)  
  

xTrain, xTest, yTrain, yTest = train_test_split(X, podaci.EC, test_size=0.20, random_state=1)

#Logistička regresija
logisticModel = LogisticRegression()
logisticModel.fit(xTrain,yTrain)

#Support vector Machine
svm = svm.SVC(kernel='rbf', C=1.0)
svm.fit(xTrain,yTrain)


yPredict1 = logisticModel.predict(xTest)
yPredict2 = svm.predict(xTest)


#Točnost modela
print("Točnost modela dobivenog logističkom regresijom:", logisticModel.score(xTest, yTest))
print("Točnost modela dobivenog SVM-om:", svm.score(xTest, yTest))

#Mean Absolute Error
print("Mean absolute error za logističku regresiju:", metrics.mean_absolute_error(yTest, yPredict1))
print("Mean absolute error za SVM:", metrics.mean_absolute_error(yTest, yPredict2))

#Mean Squared Error
print("Mean squared error za logističku regresiju:", metrics.mean_squared_error(yTest, yPredict1))
print("Mean squared error za SVM:", metrics.mean_squared_error(yTest, yPredict2))


#Root Mean Squared Error
print("Root mean squared error za logističku regresiju:", np.sqrt(metrics.mean_squared_error(yTest, yPredict1)))
print("Root mean squared error za SVM:", np.sqrt(metrics.mean_squared_error(yTest, yPredict2)))

cm1 = confusion_matrix(yTest, yPredict1)
print("Confusion matrix za model dobiven logističkom regresijom:")
print(cm1)
print("\n")

cm2 = confusion_matrix(yTest, yPredict2)
print("Confusion matrix za model dobiven SVM-om:")
print(cm2)
print("\n")

#logistička regresija
plt.figure(figsize=(9,9))
sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual');
plt.xlabel('Predicted');
all_sample_title = 'Accuracy of the model obtained by Logistic Regression: {0}'.format(logisticModel.score(xTest, yTest))
plt.title(all_sample_title, size = 15);

#svm
plt.figure(figsize=(9,9))
sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual')
plt.xlabel('Predicted')
all_sample_title = 'Accuracy of the model obtained by SVM: {0}'.format(svm.score(xTest, yTest))
plt.title(all_sample_title, size = 15)

print(yTest.sum())

