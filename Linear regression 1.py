#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 


# In[6]:


import numpy as np


# In[7]:


import sklearn


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


from sklearn.linear_model import LinearRegression


# In[13]:


df=pd.read_csv(r"F:\practice\Dataset\california_housing_train.csv")
df.head(10)


# In[14]:


df.columns


# In[15]:


df.dtypes


# In[16]:


df.shape


# In[17]:


x=df.drop(columns=['median_house_value'])


# In[19]:


x.head(10)


# In[20]:


y=df['median_house_value']


# In[22]:


y.head(10)


# In[56]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.50)


# In[57]:


reg = LinearRegression().fit(x_train,y_train)


# In[58]:


reg.coef_


# In[59]:


reg.intercept_


# In[60]:


y_pred=reg.predict(x_test)
y_pred


# In[61]:


sklearn.metrics.r2_score(y_test,y_pred)


# In[ ]:




