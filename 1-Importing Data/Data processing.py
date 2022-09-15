# Importing the libraries
import numpy as np
import pandas as pd
data_set= pd.read_csv('Dataset.csv')
x= data_set.iloc[:,:-1].values 
y= data_set.iloc[:,3].values

#Handling missing data (Replacing missing data with the mean value) 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)
print('\n')
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=1) 
print(x_train)
print(x_test)
print(y_train)
print(y_test)
print('\n')
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler() 
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
