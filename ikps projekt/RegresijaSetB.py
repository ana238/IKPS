import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge

podaci = pd.read_csv(r'C:\Users\dudic\Desktop\SetBSolcast.csv')
podaci = podaci.replace(np.nan, 0)
X=podaci[['AirTemp', 'AlbedoDaily', 'CloudOpacity', 'DewpointTemp', 'Dhi', 'Dni', 'Ebh', 'Ghi','GtiFixedTilt', 'GtiTracking', 'PrecipitableWater', 'RelativeHumidity', 'SurfacePressure', 'WindDirection10m', 'WindSpeed10m', 'Zenith' ]].values
y=podaci['ecoli'].values


#Broj stvarnih vrijednosti Escherichie coli koje prelaze 300CFU/mL
brojac = 0
for i in podaci.ecoli:
    if i >= 300:
        brojac = brojac +1  
print("Broj vrijednosti Escherichie coli koje prelaze 300 iznosi:", brojac)  
print("\n")      

#Prikaz prosječne vrijednosti ecoli odnosno ditribucija vrijednosti Escherichie coli
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(podaci['ecoli'])
plt.show()

XTreniraj, XTest, yTreniraj, yTest = train_test_split(X, y, test_size=0.2, random_state =0)

#Linearna regresija
regressor = LinearRegression()
regressor.fit (XTreniraj, yTreniraj)

#Support Vector Regression
svr = SVR(kernel='rbf', C=100000, epsilon=0.1)
svr.fit(XTreniraj,yTreniraj)

#CART
dtr = DecisionTreeRegressor()
dtr = dtr.fit(XTreniraj,yTreniraj)

#Gaussian Process Regression
gpr = GaussianProcessRegressor()
gpr.fit(XTreniraj, yTreniraj)

#Bayesian Ridge
bayRidge = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
       normalize=False, tol=0.001, verbose=False)
bayRidge.fit(XTreniraj, yTreniraj)



yPredict1 = svr.predict(XTest)
yPredict2 = regressor.predict(XTest)
yPredict3 = dtr.predict(XTest)
yPredict4 = gpr.predict(XTest)
yPredict5 = bayRidge.predict(XTest)

podaci = pd.DataFrame({'Stvarne vrijednosti': yTest, 'Predviđeno SVR-om': yPredict1, 'Predviđeno linearnom regresijom': yPredict2})
podaci1= podaci.head(50)
print(podaci1)
print("\n")

#Točne vrijednosti odstupanja predviđenih od stvarnih vrijednosti za SVM algoritam

array = []
posto = []
razlika1 = 0
razlika2 = 0
p = 0
br = 0
for i in range(0, len(yTest)):
    if yTest[i] > yPredict1[i]:
        razlika1 = (int)(yTest[i] - yPredict1[i])
        array.append(razlika1)
        razlika1 = 0
        p = (int)(yPredict1[i] * 100 / (yTest[i]))
        posto.append(p)
        pp = list(map("{}%".format, posto))
    elif yTest[i] < yPredict1[i]:
        br = br + 1
        razlika2 = (int)(yPredict1[i] - yTest[i])
        array.append(razlika2)
        razlika2 = 0
        p = (int)(yTest[i] * 100 / (yPredict1[i]))
        p = 100 - p
        posto.append(p)
        pp = list(map("{}%".format, posto))
                
print("Predviđene su vrijednosti", br, "puta prešle stvarne vrijedosti na testiranju od ", len(yTest), "vrijednosti.") 
print("\n")         
print("Razlike u vrijendostima stvarnih i predviđenih vrijednosti za SVR algoritam: ", array)
print("Razlike izražene u postocima u vrijendostima stvarnih i predviđenih vrijednosti za SVR algoritam: ", pp)


a1 = []
a2 = []
posto1 = []
posto2 = []
brojac1 = []
brojac2 = []

a11 = []
a22 = []
posto11 = []
posto22 = []
brojac11 = []
brojac22 = []
pp2 =0
for i in range(0, len(yPredict1)):
    if yPredict1[i] > 300:
        if yTest[i] > yPredict1[i]:
            r1 = (int)(yTest[i] - yPredict1[i])
            a1.append(r1)
            r1 = 0
            p1 = (int)(yPredict1[i] * 100 / (yTest[i]))
            posto1.append(p1)
            pp1 = list(map("{}%".format, posto1))
            brojac1.append(i)
        elif yTest[i] < yPredict1[i]:
            r2 = (int)(yPredict1[i] - yTest[i])
            a2.append(r2)
            r2 = 0
            p2 = (int)(yTest[i] * 100 / (yPredict1[i]))
            p2 = 100 - p2
            posto2.append(p2)
            pp2 = list(map("{}%".format, posto2))
            brojac2.append(i)
    elif yPredict1[i] < 300: 
        if yTest[i] > yPredict1[i]:
            r11 = (int)(yTest[i] - yPredict1[i])
            a11.append(r11)
            r11 = 0
            p11 = (int)(yPredict1[i] * 100 / (yTest[i]))
            posto11.append(p11)
            pp11 = list(map("{}%".format, posto11))
            brojac11.append(i)
        elif yTest[i] < yPredict1[i]:
            r22 = (int)(yPredict1[i] - yTest[i])
            a22.append(r22)
            r22 = 0
            p22 = (int)(yTest[i] * 100 / (yPredict1[i]))
            p22 = 100 - p22
            posto22.append(p22)
            pp22 = list(map("{}%".format, posto22))
            brojac22.append(i)        
            
#Kada previđena vrijednost prelazi granicu od 300CFU   
#Kada je stvarna vrijednost veća od predviđene 
print("\n")
print("1.KADA PREDVIĐENA VRIJEDNOST PRELAZI GRANICU OD 300CFU")
print("KADA JE STVARNA VRIJEDNOST VEĆA OD PREDVIĐENE:")
print("Razlike u vrijednostima stvarnih i predviđenih vrijednosti kada predviđena vrijednost prelazi granicu od 300CFU za SVR algoritam: ", a1, "za dan ", brojac1) 
print("Ukupno je takvih vrijednosti: ", len(a1))
print("Razlike izražene u postocima: ", pp1)
#Kada je stvarna vrijednost manja od predviđene
print("\n")
print("KADA JE STVARNA VRIJEDNOST MANJA OD PREDVIĐENE:")
print("Razlike u vrijednostima stvarnih i predviđenih vrijednosti kada predviđena vrijednost prelazi granicu od 300CFU za SVR algoritam: ", a2, "za dan ", brojac2) 
print("Ukupno je takvih vrijednosti: ", len(a2))
print("Razlike izražene u postocima: ", pp2)
print("\n\n") 

#Kada je predviđena vrijednost ispod granice od 300CFU
#Kada je stvarna vrijednost veća od predviđene 
print("\n")
print("1.KADA PREDVIĐENA VRIJEDNOST  NE PRELAZI GRANICU OD 300CFU")
print("KADA JE STVARNA VRIJEDNOST VEĆA OD PREDVIĐENE:")
print("Razlike u vrijednostima stvarnih i predviđenih vrijednosti kada je predviđena vrijednost ispod granice od 300CFU za SVR algoritam: ", a11, "za dan ", brojac11) 
print("Ukupno je takvih vrijednosti: ", len(a11))
print("Razlike izražene u postocima: ", pp11)
#Kada je stvarna vrijednost manja od predviđene
print("\n")
print("KADA JE STVARNA VRIJEDNOST MANJA OD PREDVIĐENE:")
print("Razlike u vrijednostima stvarnih i predviđenih vrijednosti kada je predviđena vrijednost ispod granice od 300CFU za SVR algoritam: ", a22, "za dan ", brojac2) 
print("Ukupno je takvih vrijednosti: ", len(a22))
print("Razlike izražene u postocima: ", pp22)
print("\n\n")

        
podaci = pd.DataFrame({'Stvarne vrijednosti': yTest, 'Predviđeno CART-om': yPredict3})
podaci2 = podaci.head(50)
#print(podaci2)

podaci = pd.DataFrame({'Stvarne vrijednosti': yTest, 'Predviđeno GPR-om': yPredict4})
podaci3 = podaci.head(50)
#print(podaci3)

podaci = pd.DataFrame({'Stvarne vrijednosti': yTest, 'Predviđeno Bayesian Ridge-om': yPredict5})
podaci4 = podaci.head(50)
#print(podaci4)

print("Ukupno je ", len(yTest), " podataka za testiranje i ", len(yPredict1), " podataka predviđenih vrijednosti.")

#Broj vrijednosti Escherichie coli koje prelaze granicu od 300CFU/mL u testnim vrijednostima
print("U stvarnim vrijednostima koje su uzete za testiranje ", (yTest > 300).sum(), " ih prelazi granicu od 300CFU/mL.")  
postotak = 0
postotak = (yTest > 300).sum()* 100 / (len(yTest)) 
print((yTest > 300).sum(), " vrijednosti predstavlja ", postotak, "% od ukupnog broja vrijednosti za testiranje.")     
print("\n")      
     
    

#Broj vrijednosti Escherichie coli koje prelaze granicu od 300CFU/mL u vrijednostima dobivenim SVR-om 
print("U predviđenim vrijednostima dobivenim SVR-om ", (yPredict1 > 300).sum(), " ih prelazi granicu od 300CFU/mL.")   
postotak = 0
postotak = (yPredict1 > 300).sum()* 100 / (len(yPredict1)) 
print((yPredict1 > 300).sum(), " vrijednosti predstavlja ", postotak, "% vrijednosti koje prelaze granicu od 300CFU/mL unutar previđenih vrijednosti dobivenih SVR-om.")      
print("\n")  



#Broj vrijednosti Escherichie coli koje prelaze granicu od 300CFU/mL u vrijednostima dobivenim linearnom regresijom
print("U predviđenim vrijednostima dobivenim linearnom regresijom ", (yPredict2> 300).sum(), " ih prelazi granicu od 300CFU/mL.")  
postotak = 0
postotak = (yPredict2 > 300).sum()* 100 / (len(yPredict2)) 
print((yPredict2 > 300).sum(), " vrijednosti predstavlja ", postotak, "% vrijednosti koje prelaze granicu od 300CFU/mL unutar previđenih vrijednosti dobivenih linearnom regresijom.")          
print("\n")      

#Točnost modela
print("Točnost modela dobivenog linearnom regresijom:", regressor.score(XTest, yTest))
print("Točnost modela dobivenog  metodom potpornih vektora:", svr.score(XTest, yTest))
print("Točnost modela dobivenog  CART algoritmom:", dtr.score(XTest, yTest))
print("Točnost modela dobivenog  Gausovim procesom", gpr.score(XTest, yTest))
print("Točnost modela dobivenog  Bayesian Ridge-om:", bayRidge.score(XTest, yTest))

#Mean Absolute Error
print("Mean absolute error za SVR:", metrics.mean_absolute_error(yTest, yPredict1))
print("Mean absolute error za linearnu regresiju:", metrics.mean_absolute_error(yTest, yPredict2))
print("Mean absolute error za CART:", metrics.mean_absolute_error(yTest, yPredict3))
print("Mean absolute error za GPR:", metrics.mean_absolute_error(yTest, yPredict4))
print("Mean absolute error za Bayesian Ridge:", metrics.mean_absolute_error(yTest, yPredict5))

#Mean Squared Error
print("Mean squared error za SVR:", metrics.mean_squared_error(yTest, yPredict1))
print("Mean squared error za linearnu regresiju:", metrics.mean_squared_error(yTest, yPredict2))
print("Mean squared error za CART:", metrics.mean_squared_error(yTest, yPredict3))
print("Mean squared error za GPR:", metrics.mean_squared_error(yTest, yPredict4))
print("Mean squared error za Bayesian Ridge:", metrics.mean_squared_error(yTest, yPredict5))


#Root Mean Squared Error
print("Root mean squared error za SVR:", np.sqrt(metrics.mean_squared_error(yTest, yPredict1)))
print("Root mean squared error za linearnu regresiju:", np.sqrt(metrics.mean_squared_error(yTest, yPredict2)))
print("Root mean squared error za CART:", np.sqrt(metrics.mean_squared_error(yTest, yPredict3)))
print("Root mean squared error za GPR:", np.sqrt(metrics.mean_squared_error(yTest, yPredict4)))
print("Root mean squared error za Bayesian Ridge:", np.sqrt(metrics.mean_squared_error(yTest, yPredict5)))




#graf za linearnu regresiju i SVR
podaci1.plot(kind='line',figsize=(15, 10), linewidth='3')

plt.axhline(y=300, color = 'red', linestyle='--', linewidth='2', label='Escherichia coli')
for i in range(0, 50):
    plt.axvline(i, color = "black", linestyle='--', linewidth='1')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.legend()
plt.show()

#SVR
x_ax = range(len(yTest))
plt.axhline(y=300, color = 'red', linestyle='--', linewidth=0.5, label='Escherichia coli')
plt.scatter(x_ax, yTest, s=5, color="blue", label="Original data")
plt.plot(x_ax, yPredict1, lw=0.8, color="green", label="Predicted data-SVR")
plt.legend()
plt.show() 

#Linearna regresija
x_ax = range(len(yTest))
plt.axhline(y=300, color = 'red', linestyle='--', linewidth=0.5, label='Escherichia coli')
plt.scatter(x_ax, yTest, s=5, color="blue", label="Original data")
plt.plot(x_ax, yPredict2, lw=0.8, color="green", label="Predicted data-linear regression")
plt.legend()
plt.show() 

#graf za CART
podaci2.plot(kind='line',figsize=(15, 10), linewidth='3')

plt.axhline(y=300, color = 'red', linestyle='--', linewidth='2', label='Escherichia coli')
for i in range(0, 50):
    plt.axvline(i, color = "black", linestyle='--', linewidth='1')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.legend()
plt.show()

x_ax = range(len(yTest))
plt.axhline(y=300, color = 'red', linestyle='--', linewidth=0.5, label='Escherichia coli')
plt.scatter(x_ax, yTest, s=5, color="blue", label="Original data")
plt.plot(x_ax, yPredict3, lw=0.8, color="green", label="Predicted data-CART")
plt.legend()
plt.show() 

#graf za GPR
podaci3.plot(kind='line',figsize=(15, 10), linewidth='3')

plt.axhline(y=300, color = 'red', linestyle='--', linewidth='2', label='Escherichia coli')
for i in range(0, 50):
    plt.axvline(i, color = "black", linestyle='--', linewidth='1')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.legend()
plt.show()

x_ax = range(len(yTest))
plt.axhline(y=300, color = 'red', linestyle='--', linewidth=0.5, label='Escherichia coli')
plt.scatter(x_ax, yTest, s=5, color="blue", label="Original data")
plt.plot(x_ax, yPredict4, lw=0.8, color="green", label="Predicted data-GPR")
plt.legend()
plt.show() 

#graf za Bayesian Ridge
podaci4.plot(kind='line',figsize=(15, 10), linewidth='3')

plt.axhline(y=300, color = 'red', linestyle='--', linewidth='2', label='Escherichia coli')
for i in range(0, 50):
    plt.axvline(i, color = "black", linestyle='--', linewidth='1')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.legend()
plt.show()

#prikaz predviđenih vrijednosti za cijeli set testiranja
x_ax = range(len(yTest))
plt.axhline(y=300, color = 'red', linestyle='--', linewidth=0.5, label='Escherichia coli')
plt.scatter(x_ax, yTest, s=5, color="blue", label="Original data")
plt.plot(x_ax, yPredict5, lw=0.8, color="green", label="Predicted data-Bayesian Ridge")
plt.legend()
plt.show() 