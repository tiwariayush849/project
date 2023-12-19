#!/usr/bin/env python
# coding: utf-8

# In[36]:


import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[37]:


p_dataset= pd.read_csv('lincoln.csv',sep='\t')


# In[38]:


q_dataset= p_dataset.dropna()


# In[39]:


q_dataset


# In[40]:


q_dataset['Answer'].value_counts()


# In[41]:


x=q_dataset['Question']
y=q_dataset['Answer']


# In[42]:


print(x)


# In[43]:


print(y)


# In[44]:


q_dataset[q_dataset['Question']=='Nan']


# In[66]:


x_train, x_test,y_train, y_test= train_test_split(x,y, test_size=0.1, random_state=25)


# In[67]:


x_train.shape


# In[68]:


vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


# In[69]:


model = MultinomialNB()


# In[70]:


model.fit(x_train_vectorized, y_train)


# In[71]:


predictions = model.predict(x_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


# In[51]:


classifier = make_pipeline(CountVectorizer(), LogisticRegression())
classifier.fit(x_train, y_train)

# Taking user input
input_data = input("Enter input data: ")

# Making predictions
prediction = classifier.predict([input_data])
print(prediction)


# In[ ]:





# In[ ]:




