######################## ANN Algorithm ###################################
#Library Importing
import numpy as np
import pandas as pd
import keras
#import tensorflow
import sklearn
#Data Importing

data=pd.read_csv("inputAllR01.csv")
X=data.drop(['Depth'],axis=1)
y=data['Depth']

#Data Preprocessing

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.1)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Model Initialization
from keras.layers import Dense
from keras.models import Sequential
Neural_Network=Sequential()
Neural_Network.add(Dense(units=4,activation='relu',kernel_initializer='uniform',input_dim=7))
Neural_Network.add(Dense(units=4,activation='relu',kernel_initializer='uniform'))
Neural_Network.add(Dense(units=1,activation='relu',kernel_initializer='uniform'))
Neural_Network.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

## Model Fitting and Prediction

Neural_Network.fit(X_train,y_train,batch_size=32, epochs=100)
pred=Neural_Network.predict(X_test)

##########################################################################################
#print Actual and predicted and difference
actual=np.array(y_test)

#print(result)
#print(len(X_test))
#print(pred)
#import matplotlib.pyplot as plt
#plt.plot(actual)
#plt.ylabel('some numbers')
#plt.show()
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
    outputArray[i][2]=pred[i][0]
    #outputArray[i][2]=actual[i][0]-pred[i][0]
    
np.savetxt('plotoutputAllR01.csv', outputArray, delimiter=',', fmt='%f')
#########################################################################################

from sklearn.metrics import mean_squared_error
print("\n Mean Squared Error:",mean_squared_error(y_test,pred))

## Model Saving

from keras.models import model_from_json

# serialize model to JSON
model_json = Neural_Network.to_json()
with open("Neural_Network.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
Neural_Network.save_weights("Neural_Network.h5")
print("Saved model to disk")