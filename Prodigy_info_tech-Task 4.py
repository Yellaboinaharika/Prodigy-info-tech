#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries

import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd 
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import matplotlib.pyplot as plt  
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV


# In[2]:


col_names = ['ID', 'Entity', 'Sentiment', 'Content']
df1 = pd.read_csv('twitter_training.csv', names=col_names)
df2 = pd.read_csv('twitter_validation.csv', names=col_names)
df1
df2


# In[3]:


df1.isnull().sum()


# In[4]:


#drop null values
df1.dropna(subset=['Content'], inplace=True)


# In[5]:


df1['Sentiment'] = df1['Sentiment'].replace('Irrelevant', 'Neutral')
df1['Sentiment'] = df1['Sentiment'].replace('Irrelevant', 'Neutral')


# In[6]:


#sentiment distribution

sentiment_counts = df1['Sentiment'].value_counts().sort_index()

sentiment_labels = ['Negative', 'Neutral', 'Positive']
sentiment_colors = ['red', 'grey', 'green']

fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, 
                             values=sentiment_counts.values,
                             textinfo='percent+value+label',
                             marker_colors=sentiment_colors,
                             textposition='auto',
                             hole=.3)])

fig.update_layout(
    title_text='Sentiment Distribution',
    template='plotly_white',
    xaxis=dict(
        title='Sources',
    ),
    yaxis=dict(
        title='Number of Posts in Twitter',
    )
)

fig.update_traces(marker_line_color='black', 
                  marker_line_width=1.5, 
                  opacity=0.8)
 
fig.show()


# In[7]:


# Sentiment distribution
sns.countplot(x='Sentiment', data=df1)
plt.title('Sentiment Distribution')
plt.show()


# In[8]:


# Entity and sentiment relationship
entity_sentiment_counts = df1.groupby(['Entity', 'Sentiment']).size().unstack()
entity_sentiment_counts.plot(kind='bar', stacked=True)
plt.title('Entity-Sentiment Relationship')
plt.show()


# In[9]:


#top 10 entity

top10_entity_counts = df1['Entity'].value_counts().sort_values(ascending=False)[:10]

fig = px.bar(x=top10_entity_counts.index, 
             y=top10_entity_counts.values,
             color=top10_entity_counts.values,
             text=top10_entity_counts.values,
             color_continuous_scale='Blues')

fig.update_layout(
    title_text='Top 10 Twitter Entity Distribution',
    template='plotly_white',
    xaxis=dict(
        title='Entity',
    ),
    yaxis=dict(
        title='Number of Posts in Twitter',
    )
)

fig.update_traces(marker_line_color='black', 
                  marker_line_width=1.5, 
                  opacity=0.8)
 
fig.show()


# In[10]:


#sentiment distribution in top3 entity
top3_entity_df = df1['Entity'].value_counts().sort_values(ascending=False)[:3]
top3_entity = top3_entity_df.index.tolist()
sentiment_by_entity = df1.loc[df1['Entity'].isin(top3_entity)].groupby('Entity')['Sentiment'].value_counts().sort_index()

sentiment_labels = ['Negative', 'Neutral', 'Positive']
sentiment_colors = ['red', 'grey', 'green']

row_n = 1
col_n = 3

fig = make_subplots(rows=row_n, cols=col_n, 
                    specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=top3_entity)

for i, col in enumerate(top3_entity):
    fig.add_trace(
        go.Pie(labels=sentiment_labels, 
                values=sentiment_by_entity[col].values, 
                textinfo='percent+value+label',
                marker_colors=sentiment_colors,
                textposition='auto',
                name=col),
            row=int(i/col_n)+1, col=int(i%col_n)+1)
    
fig.update_traces(marker_line_color='black', 
                  marker_line_width=1.5, 
                  opacity=0.8)

fig.show()


# In[11]:


sns.histplot(df1["Content"].str.len(), binwidth=50)
plt.show()


# In[12]:


#Count entity per category
plot1=df1.groupby(by=["Entity","Sentiment"]).count().reset_index()
plot1.head()


# In[13]:


plt.figure(figsize=(20,6))
sns.barplot(data=plot1,x="Entity",y="ID",hue="Sentiment")
plt.xticks(rotation=90)
plt.xlabel("Brand")
plt.ylabel("Number of tweets")
plt.grid()
plt.title("Distribution of tweets per Branch and sentiment");


# In[ ]:




