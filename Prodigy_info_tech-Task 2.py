#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df= pd.read_csv('Titanic-Dataset.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.describe()


# In[8]:


df.duplicated().sum()


# In[9]:


Survived = df['Survived'].value_counts().reset_index()
Survived


# In[10]:


data = {'Survived': ['Male - No', 'Male - Yes', 'Female - No', 'Female - Yes'],
        'Counts': [100, 50, 30, 80]}  # replace with actual counts
Survived = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
plt.bar(Survived['Survived'], Survived['Counts'],color=["red","blue","red","pink"])
plt.xticks(Survived['Survived'])
plt.title('Comparison of Survival')
plt.xlabel('Gender and Survival Status')
plt.ylabel('Number of People')
plt.show()


# In[11]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[12]:


inputs = df.drop('Survived',axis='columns')
target = df['Survived']


# In[13]:


sex=pd.get_dummies(inputs.Sex)
sex.head()


# In[14]:


inputs=pd.concat([inputs,sex],axis="columns")
inputs.head()


# In[15]:


inputs.drop(["Sex"],axis="columns",inplace=True)


# In[16]:


inputs.head()


# In[17]:


inputs.isna().sum()


# In[18]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()


# In[19]:


inputs.info()


# In[20]:


inputs.isna().sum()


# In[21]:


counts = df.groupby(['Survived', 'Sex']).size().unstack().fillna(0)

# Define the bar width
bar_width = 0.35
index = counts.index

# Plotting
fig, ax = plt.subplots()

# Plot bars for each Sex
bar1 = ax.bar(index - bar_width/2, counts['male'], bar_width, label='male')
bar2 = ax.bar(index + bar_width/2, counts['female'], bar_width, label='female')

# Setting labels and title
ax.set_xlabel('Survived')
ax.set_ylabel('Count')
ax.set_title('Survival Counts by Gender')
ax.set_xticks(index)
ax.set_xticklabels(['Not Survived', 'Survived'])
ax.legend()

# Display the plot
plt.show()


# In[22]:


X_train, X_test, y_train, y_test=train_test_split(inputs,target,test_size=0.2)


# In[23]:


X_train


# In[24]:


X_test


# In[25]:


y_train


# In[26]:


y_test


# In[27]:


inputs.corr()


# In[28]:


import seaborn as sns


# In[29]:


sns.heatmap(inputs.corr(), annot=True, cmap='coolwarm', fmt=".2f")


# In[30]:


model=RandomForestClassifier()


# In[31]:


model.fit(X_train,y_train)


# In[32]:


model.score(X_test,y_test)


# In[33]:


pre=model.predict(X_test)


# In[34]:


matrices=r2_score(pre,y_test)
matrices


# In[ ]:




