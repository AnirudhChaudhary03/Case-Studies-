# Calling Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
###############################
data=pd.read_csv('D:/ANIRUDH/DATA SCIENCE PROJECT/1.Insurance case study/insurance.csv') # Data Extraction
###############################
data.columns # Checking Columns
data.info() # Checking Information like null or non-null and object or data type
data.describe() # Brief Describtion like mean,count,std,min,max
data.isna().sum() # Finding zero Values and its Count
###############################
# Use Scatter Plot : Numerical - Numerical
# Use Bar and Box Plot : Catagorical - Numerical
###############################
sns.scatterplot(x=data['age'],y=data['charges'])
sns.scatterplot(x=data['bmi'],y=data['charges'])
###############################
#Gender vs Charges
sns.boxplot(x=data['sex'],y=data['charges'])
#Children vs Charges
sns.boxplot(x=data['children'],y=data['charges'])
###############################
#Smoker vs Charges
sns.boxplot(x=data['smoker'],y=data['charges'])
#Region vs Charges
sns.boxplot(x=data['region'],y=data['charges'])
############################### 
# Counting Unique Values And its Count
columns=['sex','smoker', 'region']
for column in columns:
    print(data[column].unique())
    print(data[column].value_counts())
###############################
# Converting Categorical Values into Numberical 
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder() 
###############################
columns=['sex','smoker', 'region']
for column in columns:
    data[column]=encoder.fit_transform(data[column])
###############################
# Correlation btw the two and more parameters
sns.heatmap(data.loc[:,('age','bmi','smoker','children','charges')].corr(),annot=True)
###############################
x=data.drop(['charges'],axis=1) # Assigning Input parameters to x
y=data['charges'] # Assigning Output parameters to y
###############################
# Splitting Data for Training and Testing 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
############################### 
# Create Model and Selecting the important Columns 
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
sel=SelectFromModel(Lasso(alpha=0.05))
sel.fit(x_train,y_train)
sel.get_support()
###############################
x.columns[sel.get_support()] # Checking Lasso Columns
###############################
x_train=x_train.loc[:,['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
x_test=x_test.loc[:,['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
#          LASSO
from sklearn.linear_model import Lasso
regressor1=Lasso(alpha=0.05) # Default Value for Alpha : 0.05
regressor1.fit(x_train,y_train)
###############################
regressor1.coef_
regressor1.intercept_
###############################
y_pred1=regressor1.predict(x_test) # Predicted Value By Lasso
###############################
from sklearn import metrics # Accuracy Checking
np.sqrt(metrics.mean_squared_error(y_test, y_pred1)) # 5663.3634366773895
metrics.mean_absolute_error(y_test, y_pred1) # 3998.2818378379643
metrics.r2_score(y_test, y_pred1) # 0.7962728448333078
###############################
#           RIDGE
from sklearn.linear_model import Ridge
regressor2=Ridge(alpha=0.09)
regressor2.fit(x_train,y_train)
###############################
regressor2.coef_
regressor2.intercept_
###############################
y_pred2=regressor2.predict(x_test) # Predicted Value By Ridge
###############################
from sklearn import metrics # Accuracy Checking
np.sqrt(metrics.mean_squared_error(y_test, y_pred2)) # 5663.700927923505
metrics.mean_absolute_error(y_test, y_pred2) # 3999.5074960154316
metrics.r2_score(y_test, y_pred2) # 0.7962485630859357
###############################
from sklearn.linear_model import ElasticNet # Lasso and Ridge Regression 
regressor3=ElasticNet(alpha=0.09)
regressor3.fit(x_train,y_train)
###############################
regressor3.coef_
regressor3.intercept_
###############################
y_pred3=regressor3.predict(x_test) # Predicted Value By Lasso and Ridge Both
###############################
from sklearn import metrics # Accuracy Checking
np.sqrt(metrics.mean_squared_error(y_test, y_pred3)) # 6170.941188002623
metrics.mean_absolute_error(y_test, y_pred3) # 4605.667991869292
metrics.r2_score(y_test, y_pred3) # 0.7581183790594637
# ------------- END ------------