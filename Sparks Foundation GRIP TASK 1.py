#!/usr/bin/env python
# coding: utf-8

# Sparks Foundation GRIP TASK 

# Task 1
# 
# Simple Linear Regression
# In this regression task we will predict the percentage of marks that a student 
# is expected to score based upon the number of hours they studied. 
# This is a simple linear regression task as it involves just two variables.
# 
# Steps to be followed: 
# 1.Importing the data
# 2.Visualizing the data
# 3.Data Preparation
# 4.Training the model
# 5.Visualizing the model
# 6.Making predictions
# 7.Evaluating the model
# 
# 

# Name: Amritangshu Bandyopadhyay 

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import os
import seaborn as sns
import matplotlib.pyplot as plt  


# In[2]:


url= "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data=pd.read_csv(url)
print ("Imported data successfully")


# In[3]:


s_data.head(10)


# In[5]:


s_data.info()


# In[7]:


s_data.describe()


# In[8]:


s_data.hist()


# Plotting the data on a 2D graph
# 

# In[10]:


s_data.plot(x='Hours', y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage score')
plt.grid()
plt.show()


# There is a positive linear relationship between the two variables plotted

# Preparing the Data 

# In[12]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# Training the algorithm 

# In[13]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[14]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# Making Predictions 

# In[15]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[16]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[17]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# We assume the following for a random prediction:
# No of Hours = 9.25
# Predicted Score = 93.69173248737539
# Therfore, the predicted score if a student studies for 9.25 hrs/day is about 93.69.

# Evaluating the model

# In[18]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# This concludes Task 1
# Thank you!

# In[ ]:




