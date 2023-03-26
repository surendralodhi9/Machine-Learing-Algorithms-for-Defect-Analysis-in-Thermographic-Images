import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import StandardScaler
   
# load the digit dataset 
digits = datasets.load_digits()
data=pd.read_csv("input.csv")
X=data.drop(['Depth'],axis=1)
y=data['Depth']

#print(X)
#print(y)
#print(digits)
   
# defining feature matrix(X) and response vector(y) 
#X = digits.data 
#y = digits.target

#print(len(X))
#print(len(y))
  
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=1)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)


print(X_test)
scaler = StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)


   
# create logistic regression object 
reg = linear_model.LogisticRegression() 
   
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = reg.predict(X_test)

print(y_pred)
#print(len(y_test))
   
# comparing actual response values (y_test) with predicted response values (y_pred) 
print("Logistic Regression model accuracy(in %):",  
metrics.accuracy_score(y_test, y_pred)*100) 

pred=y_pred
actual=np.array(y_test)
actual=np.reshape(actual,(-1,1))
#####################################
outputArray = [[0 for i in range(3)] for j in range(len(actual))] 


for i in range(0,len(actual)-1):
    
    #print(actual[i][0],"-",pred[i][0],"=",actual[i][0]-pred[i][0])
    if i==0:
        
        outputArray[i][0]="Id";
        outputArray[i][1]="Actual"
        outputArray[i][2]="Predicted"
    outputArray[i][0]=i+1;
    outputArray[i][1]=actual[i][0]
    outputArray[i][2]=pred[i]
    #outputArray[i][2]=actual[i][0]-pred[i][0]
    
np.savetxt('plotoutputlogis.csv', outputArray, delimiter=',', fmt='%f')
#########################################################################################
