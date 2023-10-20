#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:





# In[ ]:





# ### NAME : SUGALI DHANUNJAYA(S.DHANU-G27 PYTHON WITH MACHINE LEARING)
# 
# #
# 

# In[ ]:





# # PROBELM STATEMENTS
You have the 'bank-full.csv' dataset, and your task is to build a machine learning model to predict the 'y' column using these classification algorithms: Logistic Regression, Naive Bayes, SVC Classifier, Decision Tree Classifier, and Random Forest Classifier. Summarize the key steps:

Data Handling: How do you load the dataset and designate 'y' as the target variable?

Data Cleaning: Describe how you'd remove useless columns, eliminate those with unique values, and handle missing data.

Data Visualization: What visualizations would you use for numerical and categorical data?

Data Analysis: Explain univariate, bivariate analysis, and any insights related to the business.

Data Preprocessing: Do you need label encoding, one-hot encoding, standardization, or normalization?

Model Implementation: Briefly outline how you'd implement the mentioned classification algorithms.

Model Validation: Mention the metrics used, especially the R2 score interpretation.

Model Comparison: How would you compare R2 scores and execution speeds to choose the best model?
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


# Importing required libraries
import pandas as pd
import numpy as np
# Data Visualization libraries
from matplotlib import pyplot as plt
import seaborn as sbn



# # DATA HANDLING

# In[2]:


#loading Date-Set
# loading the csv file by using pandas
df=pd.read_csv(r"C:\Users\dhanu\OneDrive\Documents\bank-full.csv")
df

separate each attritubute with thier corresponding data in separate columns
# In[3]:


df.shape


# In[4]:


path=(r"C:\Users\dhanu\OneDrive\Documents\bank-full.csv")
df=pd.read_csv(path,sep=';')
print(df)


# In[5]:


df.columns


# In[6]:


df.shape

understaing data setinput variable are:['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome']
output variable is :['y']  only 
    By analysing all the past data input variables to predict the result of target variable 'y' and is target variable consist of either 'Yes'-means the person who ready to subscribe or 'No'- for not ready to take subcription from the bank.   
    
# # DATA CLEAING

# In[7]:


# Information about the datatypes and their nullvalue counts of each attribute in the given dataset.

df.info()


# In[8]:


# describing the dataset to summarize each attribute

df.describe()


# # EXPLORATORY DATA ANALYSIS
THE PROCESS OF DESCRIBING  THE DATA BY STATISTICAL AND VISUALIZARIONS TECHNIQUES IN ORDER TO BRING IMPORTANT
ASPECTS OF THAT DATA INTO FOCUS FOR ANALYSIS
# In[9]:


# checking the missing values in the given dataset

df.isna().sum()


# In[10]:


# Cheaking how many  of numbers unknown terms in the given dataset for each attribute

for column in df.columns:
    print(f'Column: {column}')
    print(df[column].value_counts())
    print()


# view as a scrollable  or open in text editor

# In[11]:


#Replace_method for "unknown" term in ["job", "education", "contact"] with thier mode terms.
    
df["job"].replace(["unknown"],df["job"].mode(), inplace=True)
df["education"].replace(["unknown"],df["education"].mode(), inplace=True)
df["contact"].replace(["unknown"],df ["contact"].mode(), inplace=True)


# In[12]:


df['job'].value_counts()


# In[13]:


df["education"].value_counts()


# In[14]:


df["contact"].value_counts()


# In[15]:


# Finding categorical variables

categorical_features=[feature for feature in df.columns if (df[feature].dtypes=='O')]
print(f'There are {len(categorical_features)} categorical variables')
categorical_features


# In[16]:


df[categorical_features].head()


# In[17]:


# Finding numerical arrtributes

numerical_features = [feature for feature in df.columns if (df[feature].dtypes != 'O' )]
print(f'There are {len(numerical_features)} numerical variables')
numerical_features


# In[18]:


# Checking the head of the numerical arrtributes
df[numerical_features].head()


# # DATA VISUALIZATION AND DATA ANALYSIS

# In[19]:


# Checking the head of the numerical arrtributes
df[numerical_features].head()


# In[20]:


# Let us see the target variable 'y' through histplot

sbn.histplot(df.y)


# In[21]:


#Univariate Analysis of Categorical attributes.

plt.figure(figsize=(10,80),facecolor='SKYBLUE')
plotnumber=1
for cat_feature in categorical_features:
    x=plt.subplot(12,3,plotnumber)
    sbn.countplot(y=cat_feature,data=df)
    plt.xlabel(cat_feature)
    plt.title(cat_feature)
    plotnumber+=1
plt.show()  


# In[22]:


# ploting a boxplot for each numerical arrtribute to check the outliers

plt.figure(figsize=(10,90), facecolor='SKYBLUE')
plotnumber = 1
for num_feature in numerical_features:
    ax = plt.subplot(12,3,plotnumber)
    sbn.boxplot(df[num_feature])
    plt.xlabel(num_feature)
    plotnumber += 1
plt.show()


# In[23]:


# Distibution plot of Numerical attributes.()
plot_num=1
plt.figure(figsize=(10,80))
for i in numerical_features:
  plt.subplot(12,3,plot_num)
  plot_num= plot_num+1
  sbn.distplot(df[i],color='blue')
  plt.title(i)
  plt.tight_layout


# view as a scrollable or open in text editor

# In[24]:


# count plot of the days with the target variable 'y'

fig, ax = plt.subplots(figsize=(15, 10))
sbn.countplot(x='day', hue='y', data=df, palette='bright', ax=ax)

# labeling the plot

plt.xlabel('Day')
plt.ylabel('Count')
plt.title('Count of Days with Target Variable')
plt.show()


# In[25]:


# Bivariate Analysis of Numerical arrtributes with Target Variable

# let as see the count subscriptions rates for bank campaign on the basis of different type of job in the dataset

plt.figure(figsize=(20,10))
sbn.countplot(data=df, x='job', hue='y', palette='bright')
plt.xlabel('Job')
plt.title('Count of different job types')
plt.show()


# In[26]:


# count of people with different marital statuses taking subscription rates

plt.figure(figsize=(12, 6))
sbn.countplot(data=df, x='marital', hue='y', palette='bright')
plt.xlabel('Marital Status')
plt.title('Count of people with different marital statuses')
plt.show()


# In[27]:


# plotting the count plot of education  with target variable.

plt.figure(figsize=(12, 6))
sbn.countplot(data=df, x='education', hue='y', palette='bright')
plt.xlabel('Education Level')
plt.title('count of "NO" and "YES" for different level of education')
plt.show()


# In[28]:


# ploting count plot on the basis of month with the target variable

plt.figure(figsize=(10, 5))
sbn.countplot(data=df, x='month', hue='y', palette='bright')
plt.xlabel('Month')
plt.title('Count on the month of basis')
plt.show()


# In[29]:


# plotting the count plot of  contact type with target variable.

plt.figure(figsize=(10, 5))
sbn.countplot(data=df, x='contact', hue='y', palette='bright')
plt.xlabel('Contact Type')
plt.title('Count of "Yes" and "No" for different contact types')
plt.show()


# In[30]:


# Finding categorical arrtributes

for column in df.select_dtypes(include='object').columns:
    print(column)
    print(df[column].unique())


# I used label encoding and standardization methods to process the given data

# In[31]:


#Label encoding
from sklearn.preprocessing import LabelEncoder
columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
lab = LabelEncoder()
for column in columns_to_encode:
    df[column] = lab.fit_transform(df[column])


# In[32]:


df.head()


# In[33]:


# Let us see the target variable 'y' through histplot

sbn.histplot(df.y)


# standardization

# 
# Data standardization is important in data preprocessing because it helps to improve the accuracy and performance of machine learning models. By standardizing the data, all of the features are on the same scale, which makes it easier for the model to learn and make predictions. Many machine learning algorithms are sensitive to the scale of input arrtributes. Standardization scales arrtributes to have a mean of 0 and a standard deviation of 1, ensuring that they have similar scales and don't dominate the model's learning process. but normalization technique used to scale numerical features to a specific range while preserving their original distribution.So, here standardization technique was suitable to apply for the given data.
# 
# implementing the SMOTE to oversampling the imbalanced data

from collections import Counter
from imblearn.over_sampling import SMOTENC

from sklearn.utils._param_validation import _MissingValues

# Create the instance of SMOTE

sm = SMOTENC(random_state=0)

# Fit and transform the training data using SMOTE
X_train_os, y_train_os = sm.fit_resample(X_train, y_train)

# Calculate class distribution before and after SMOTE
original_class_distribution = Counter(y_train)
resampled_class_distribution = Counter(y_train_os)

# Print the class distribution
print('Original class distribution:', original_class_distribution)
print('Resampled class distribution:', resampled_class_distribution)

# In[ ]:





# In[35]:


#standardize the arrtributes
from sklearn.preprocessing import StandardScaler
arrtributes = df.drop("y", axis = 1) 
target_variable = df["y"] 
features_num = arrtributes.columns 
scaler = StandardScaler()
arrtributes = pd.DataFrame(scaler.fit_transform(arrtributes)) 
arrtributes.columns = features_num

arrtributes.head()


# train, test and splitting the data

# In[65]:


# splitting the data into training and testing

from sklearn.model_selection import train_test_split
y = df["y"]
X = df.drop("y",axis = 1)
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.5, random_state = 67, stratify=y)


# Logistic Regression Model
# 

# Model Implementation:
# 

# Logistic Regression, Naive Bayes, SVC Classifier, Decision Tree Classifier and Random Forest Classifier. Summarize the key steps:

# In[66]:


# Import required libraries for machine learning models

import time
import sklearn.metrics as metrics
from sklearn.metrics import r2_score, confusion_matrix, classification_report,accuracy_score


# Logistic Regression Model
# 

# 
# Logistic regression is a statistical model that predicts the probability of a binary outcome (yes/no, 0/1, or true/false) based on one or more independent variables. It is commonly used in machine learning for classification tasks.
# 
# applications : whether an email is spam or not.

# In[39]:


# Importing the required libraries for the Logistic Regression model

from sklearn.linear_model import LogisticRegression

# Creating model object for logistic regression.

model = LogisticRegression(fit_intercept=True, max_iter=10000)

# fitting the model to the trained data.

start_time = time.time()
ab=model.fit(X_train, y_train)
end_time = time.time()
ab


# In[40]:


# Calculate the training time

training_time = end_time - start_time
train_time1=(f"Training time: {training_time} seconds")
train_time1


# In[77]:


# Getting the predicted classes for testing set

start_time = time.time()
y_pred = ab.predict(X_test)
end_time = time.time()





# In[78]:


# Calculate the testing time
testing_time = end_time - start_time
test_time1=(f"Testing time: {testing_time} seconds")
test_time1


# In[42]:


# printing th confusion matrix

print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred))
r2_score1=("r2_score :",metrics.r2_score(y_test, y_pred))
r2_score1


# In[43]:


accuracy_score1=("Accuracy_score :",metrics.accuracy_score(y_test, y_pred))
accuracy_score1


# Naive Bayes Model
# 

# Naive Bayes is a simple classification algorithm that uses Bayes' Theorem to calculate the probability of an object belonging to a particular class.
# 
# The simple form of the calculation for Bayes Theorem is as follows:
# 
# P(A|B) = P(B|A) * P(A) / P(B)
# 
# applications : spam filtering, recommendation systems, and text classification.

# In[44]:


#Naive Bayes algorithm
# importing required libraries

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# Defining Grid Search Parameters

param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}

# Using GridSearchCV for hyperparameter tuning

model = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, cv=10, n_jobs=-1)

# Fitting the model

start_time = time.time()
av=model.fit(X_train, y_train)
end_time = time.time()
av


# In[45]:


# Calculate the training time

training_time = end_time - start_time
train_time2=(f"Training time: {training_time} seconds")
train_time2


# In[46]:


# Getting the predicted classes for testing set

start_time = time.time()
y_pred = av.predict(X_test)
end_time = time.time()

# Calculate the testing time

testing_time = end_time - start_time
test_time2=(f"Testing time: {testing_time} seconds")
test_time2


# In[47]:


# printng the confusion matrix

print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred))
r2_score2=("r2_score:",metrics.r2_score(y_test, y_pred))
r2_score2


# In[48]:


accuracy_score2=("Accuracy_score :",metrics.accuracy_score(y_test, y_pred))
accuracy_score2


# In[49]:


#SVC Classifier algorithm
# importing the required libraries

from sklearn.svm import SVC

# Create the support vector classification model and Fit the model to the training data

start_time = time.time()
model = SVC(kernel='poly').fit(X_train, y_train)
end_time = time.time()


# In[50]:


# Calculate the training time

training_time = end_time - start_time
train_time3=(f"Training time: {training_time} seconds")
train_time3


# In[51]:


# Getting the predicted classes for testing set

start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

# Calculate the testing time

testing_time = end_time - start_time
test_time3=(f"Testing time: {testing_time} seconds")
test_time3


# In[52]:


# printing the confusion matrix

print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test,y_pred))
r2_score3=("r2_score:",metrics.r2_score(y_test,y_pred))
r2_score3


# In[53]:


# metric accuracy_score
accuracy_score3=("Accuracy_score :",metrics.accuracy_score(y_test, y_pred))
accuracy_score3


# Decision Tree Classification Model
# 

# Decision trees are a type of machine learning model that can be used for both classification and regression problems. They work by splitting the data into smaller and smaller subsets based on the values of the features, until each subset contains only data points of the same class or with similar values for the target variable.

# In[54]:


# decision tree classifier
# Importing required libraries

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Creating model object for Decision Tree Classifier and Fitting the model.

start_time = time.time()
model = DecisionTreeClassifier().fit(X_train, y_train)
start_time = time.time()


# In[55]:


# Calculate the training time

training_time = end_time - start_time
train_time4=(f"Training time: {training_time} seconds")
train_time4


# In[56]:


# Getting the predicted classes for testing set

start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

# Calculate the testing time

testing_time = end_time - start_time
test_time4=(f"Testing time: {testing_time} seconds")
test_time4


# In[57]:


# printing the confusion matrix

print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred))
r2_score4=("r2_score : ",metrics.r2_score(y_test, y_pred))
r2_score4


# In[58]:


accuracy_score4=("Accuracy_score :",metrics.accuracy_score(y_test, y_pred))
accuracy_score4


# Random Forest Classification Model
# 

# In[59]:


# Random Forest Classifier

import time  # to check the excusion speeds of a model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# fitting the model

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()


# In[60]:


# Calculate the training time

training_time = end_time - start_time
train_time5=(f"Training time: {training_time} seconds")
train_time5


# In[61]:


# Getting the predicted classes for testing set

start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

# Calculate the testing time

testing_time = end_time - start_time
test_time5=(f"Testing time: {testing_time} seconds")
test_time5


# In[62]:


# Importing neccesary libraries

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report, r2_score

# printing the confusion matrix

print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred))
r2_score5=("r2_score ",r2_score(y_test, y_pred))
r2_score5


# In[63]:


# accuracy metric score

accuracy_score5=("Accuracy_score :",metrics.accuracy_score(y_test, y_pred))
accuracy_score5


# Model Validation:
# 

# Model Validation: Mention the metrics used, especially the R2 score interpretation

# The choice of metric depends on the type of problem you are solving (regression or classification) and the specific goals and requirements of our project. It's often advisable to use a combination of metrics to gain a more comprehensive understanding of our model's performance. Model validation is a crucial step in assessing the performance and reliability of machine learning models. Several metrics are commonly used for model validation for this given data are R-squared (R2) metric score accuracy_score Accuracy, Precision, Recall, F1-Score are used for validation of a model for this given data
# 
# R-squared (R2) Score:
# 
# The R2 score, also known as the coefficient of determination, is a metric used for regression models. It measures the proportion of the variance in the dependent variable that is explained by the independent variables in the model. The R2 score ranges from 0 to 1, with higher values indicating a better fit of the model to the data.
# 
# Interpretation of R2 score metrics:
# 
# if R2 = 0: The model explains none of the variance in the target variable, and it performs no better than simply predicting the mean of the target variable. if 0 < R2 < 1: The model explains a portion of the variance, and a higher R2 score indicates a better fit. For example, an R2 of 0.8 means that 80% of the variance in the target variable is explained by the model. R2 = 1: The model explains all of the variance in the target variable, and it is a perfect fit to the data.
# 
# accuracy_score metrics :
# 
# The accuracy_score metric is a measure of how well a machine learning model is performing on a holdout dataset. It is calculated by dividing the number of correct predictions by the total number of predictions.
# 
# For example, if a model makes 100 predictions and 80 of them are correct, then the accuracy_score would be 0.80.

# Model Comparison:
# 

#   Model Comparison: How would you compare R2 scores and execution speeds to choose the best model?

# In[80]:


# Create an empty DataFrame
scores_df = pd.DataFrame()
Model=['Logistic Regression', 'Naive Bayes', 'SVC Classifier', 'Decision Tree Classifier', 'Random Forest Classifier']
# Add scores to the DataFrame
scores_df['Model'] = ['Logistic Regression', 'Naive Bayes', 'SVC Classifier', 'Decision Tree Classifier', 'Random Forest Classifier']
scores_df['r2_score'] = [r2_score1,r2_score2,r2_score3,r2_score4,r2_score5]
scores_df['accuracy_score'] = [accuracy_score1,accuracy_score2,accuracy_score3,accuracy_score4,accuracy_score5]
scores_df['Training time'] = [train_time1,train_time2,train_time3,train_time4,train_time5]
scores_df['Testing time'] = [test_time1,test_time2,test_time3,test_time4,test_time5]


# In[67]:


# Display the DataFrame
scores_df


# In[70]:


r2_score(y_test,y_pred)
accuracy_score(y_test,y_pred)


# Observation :
# 
# By comparing all the given machine learning models we would like to choose "Random Forest Classifier Model" for the reason that it possess high Accuracy_score about 0.90 means 90% of predictions are correct, and r2_score is between 0 to 1 for this model and this score is higher for this model when compare to other models even through the excusion speeds are high when compare with other models including their scores. Execution speed refers to how quickly the model can make predictions or perform training.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




