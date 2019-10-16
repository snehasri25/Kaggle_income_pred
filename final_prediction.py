# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:02:24 2019

@author: SnehaSri
"""
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#reading training data
dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv',index_col='Instance')
print(dataset.head(10))
print(dataset.info())

#rename column
dataset = dataset.rename(index=str, columns={"Income in EUR": "Income"})

# concat testing data with training data
dt = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv',index_col='Instance')
dataset=pd.concat([dataset,dt], sort=False)
print(dataset)

#rename columns
dataset.rename(columns={
        'Body Height [cm]': 'body_height_cm', 
    }, inplace=True)
#changing titles of columns to lower case
dataset.columns = [col.lower() for col in dataset]
print(dataset.columns)

#checking for missing values in each column
print(dataset.isnull().sum())

# imputing null and missing values in year column
year= dataset['year of record']
year.head()
year_mean = year.mean()
year_mean
# filling null values with mean
year.fillna(year_mean, inplace=True)
dataset.isnull().sum()
# imputing missing values in age col
age_rec= dataset['age']
age_rec.head()
age_mean = age_rec.mean()
age_mean
# filling out null values with mean
age_rec.fillna(age_mean,inplace=True)
dataset.isnull().sum()
income_rec = dataset['income']
income_mean = income_rec.mean()
income_rec.fillna(income_mean,inplace=True)
# fill profession, university degree and hair color as 'unknown'
val = 'unknown'
prof = dataset['profession']
prof.fillna(val,inplace=True)
deg = dataset['university degree']
deg.fillna(val,inplace=True)
hair = dataset['hair color']
hair.fillna(val,inplace=True)
gender_rec = dataset['gender']
gender_rec.fillna(val,inplace=True)
dataset.isnull().sum()
df = dataset

# target encoding
df['gender'] = df['gender'].map(df.groupby('gender')['income'].mean())
df['profession'] = df['profession'].map(df.groupby('profession')['income'].mean())
df['university degree'] = df['university degree'].map(df.groupby('university degree')['income'].mean())
df['wears glasses'] = df['wears glasses'].map(df.groupby('wears glasses')['income'].mean())
df['hair color'] = df['hair color'].map(df.groupby('hair color')['income'].mean())
df['country'] = df['country'].map(df.groupby('country')['income'].mean())

#loading X and y
X = df[['year of record', 'gender', 'age', 'country', 'size of city',
       'profession', 'university degree', 'wears glasses', 'hair color',
       'body_height_cm']].values
y = df['income'].values

# fitting linear regression model
X1 = X[0:111994]
y1 = y[0:111994]
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
# running regression
y_pred = regressor.predict(X_test)
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual=",y_test)
print("Predicted=",y_pred)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# prediction on actual test data
X2 = X[111993:]
print(len(X2))
y2 = regressor.predict(X2)
print("Predicted=",y2)
n = len(y2)
print("length of y2=",n)
# writing output in a text file
with open("out_kaggle.txt","w") as f:
    for i in y2:
        f.write('%s\n'%str(i))

