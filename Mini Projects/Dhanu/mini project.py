#!/usr/bin/env python
# coding: utf-8

# 
# Mini Project Analysis
Name:Sugali Dhanunjaya
Batch:G27python
# In[ ]:


problem statement


# Using the provided dataframe containing student marks:
# 
# Data Export: Save this dataframe to a CSV file named "student_marks.csv." Make sure to include the index in the CSV file.
# 
# Data Preprocessing: After saving the data to the CSV file, load it back into a new dataframe. Perform the following preprocessing steps:
# 
# Rename the columns as follows:
# Rename "Student_ID" to "StudentID"
# Rename "Name" to "StudentName"
# Rename all "Subject_X" columns to "SubjectX" (where X is the subject number).
# Handle missing values:
# Replace missing values in numerical columns (e.g., SubjectX) with the mean value of that column.
# Analyze the data:
# Calculate and display the summary statistics for each subject (mean, median, standard deviation, min, max).
# Plot a histogram for one of the subjects to visualize the distribution of marks.
# Machine Learning Model: After preprocessing the data, build a machine learning model to predict a student's performance in "SubjectX" based on the other subjects' marks. Choose an appropriate machine learning algorithm, split the data into training and testing sets, train the model, and evaluate its performance using an appropriate metric (e.g., mean squared error for regression).
# 
# 
# Please provide the code and explanations for each step.

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


student= 80
subjects = np.random.randint(6, 9)  

data = {
    'Class': [],
    'Student_ID': [],
    'Name': []
}
for i in range(1,student + 1):
    data['Class'].append('Class A')
    data['Student_ID'].append(i)
    data['Name'].append(f'Student_{i}')
    for j in range(1, subjects + 1):
        data[f'Subject_{j}'] = np.random.randint(0, 101, student)


df = pd.DataFrame(data)

for _ in range(30):  
    row_idx = np.random.randint(0, student)
    col_idx = np.random.randint(3, df.shape[1])  
    df.iat[row_idx, col_idx] = np.nan

df


# DATA EXPORT

# In[3]:


df.to_csv('student.csv', index=True)


# In[4]:


df = pd.read_csv('student.csv')

df.rename(columns={'Student_ID': 'StudentID', 'Name': 'StudentName'}, inplace=True)
df.columns = df.columns.str.replace('Subject_', 'Subject')
df.info()


# In[5]:


df["Subject1"].fillna(df["Subject1"].mean(),inplace=True)
df["Subject2"].fillna(df["Subject2"].mean(),inplace=True)
df["Subject3"].fillna(df["Subject3"].mean(),inplace=True)
df["Subject4"].fillna(df["Subject4"].mean(),inplace=True)
df["Subject5"].fillna(df["Subject5"].mean(),inplace=True)
df["Subject6"].fillna(df["Subject6"].mean(),inplace=True)


# In[6]:


student_marks = pd.read_csv("student.csv")
student_marks.head()


# DATA PREPROCESSING

# In[7]:


student_marks.dtypes


# In[8]:


#Class and Name are Object type and these are not useful for us
student_marks.drop(["Class","Name"], axis='columns',inplace=True)


# In[9]:


student_marks.head()


# In[10]:


df.rename(columns={df.columns[0]:"Index"},inplace=True)
df


# In[11]:


df.isna().sum()


# In[12]:


df.describe()


# In[13]:


df.drop("Index",axis=1,inplace=True)
df.drop("Class",axis=1,inplace=True)
df.drop("StudentID",axis=1,inplace=True)
df.drop("StudentName",axis=1,inplace=True)
df


# In[14]:


df.isnull().sum()


# In[15]:


summary_stats = df.describe()

import matplotlib.pyplot as plt
plt.hist(df['Subject1'], bins=10, edgecolor='k')
plt.title('Distribution of Marks for Subject1')
plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.show()


# In[16]:


X=df.drop('Subject1', axis=1)
X


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[18]:


y=df["Subject1"]
y


# MODEL SELECTION

# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[20]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[21]:


print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)


# In[22]:


X_train.shape


# In[23]:


y_train.shape


# In[24]:


X_test.shape


# In[25]:


y_test.shape


# In[31]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor()
model = hgb.fit(X_train, y_train)


# In[32]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(X_train,y_train)
model


# # In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[34]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor()
model = hgb.fit(X_train, y_train)


# In[ ]:





# In[42]:


X_test = X_test.dropna() 


# In[43]:


y_pred=lr.predict(X_test)
print(y_pred)
print(y_test)


# In[44]:


student_marks.describe()


# DATA VISUALIZATION

# In[48]:


import matplotlib.pyplot as plt
plt.hist(student_marks["Subject_1"], bins=20,edgecolor="k")
plt.title(f'Histogram of Subject_1')
plt.xlabel('Marks')
plt.show()


# In[49]:


import seaborn as sns
sns.boxplot(student_marks['Subject_1'])


# In[50]:


import seaborn as sns
sns.pairplot(student_marks)


# In[63]:


summary_stats = df.describe()

import matplotlib.pyplot as plt
plt.hist(df['Subject1'], bins=10, edgecolor='k')
plt.title('Distribution of Marks for Subject1')
plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.show()


# In[ ]:




