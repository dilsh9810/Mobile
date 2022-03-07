# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


#Load the dataset
df = pd.read_csv("Gardening Crops.csv")

df = df.iloc[:,:].values
df1 = pd.DataFrame(df)


#Get first 5 rows of the dataset
df1.head()
df1.info()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


Waterlevel = LabelEncoder()

df[:,4] = Waterlevel.fit_transform(df[:,4])

df1 = pd.DataFrame(df)

x = df1.iloc[:,0:6]
y = df1.iloc[:,-1]

#Split the features and target variable of the dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1)

#Display train and test values

print(x_train)
print(y_test)
print(x_test)

print(x_train.dtypes)
print(x_test.dtypes)
#print(df.dtypes)


#Build and train the DecisionTreeClassifier model

clf = DecisionTreeClassifier()

clf.fit(x_train, y_train)

Predict = clf.predict(x_test)

print(Predict)

accuracy = accuracy_score(y_test, Predict)

print('Model Training Finished.\n\tAccuracy obtained: {}'.format(accuracy * 100))

#print(classificstion_report(y_test,Predict))
#save the python model in the disk before write the object of python model in separate file
import  pickle

pickle.dump(clf,open('crop-model.model.pkl','wb'))













