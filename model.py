import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#reading the file
data=pd.read_csv('C:/Users/lenovo/Desktop/ProjectS/ML & DL/ML projects/Placement-Predictor-Flask/Placement_Data_Full_Class.csv - Sheet1.csv')
data

#droping the serial no and salary col
data=data.drop('sl_no',axis=1)
data=data.drop('salary',axis=1)
data.head()

# for i in range(0,215):
#     if(data['gender'][i]=='M'):
#         data['gender'][i]=1
#     else:
#         data['gender'][i]=0


#catgorising col for further labelling
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes

#labelling the col
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data

#selecting the features and labels
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
Y

#dividing the data into train and split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_train
data.head()

#creating a classifier using sklearn
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train,Y_train)
#printing the acc
clf.score(X_test,Y_test)


#predicting for random value
clf.predict([[1,77,0,95,0,2,80,2,0,0,1,0]])


#creating a Y_pred for test data
Y_pred=clf.predict(X_test)
Y_pred

#model generation
pickle.dump(clf, open('model.pkl','wb'))
#evalution of the classifier
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy_score(Y_test, Y_pred)