#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('data.csv')
data.head


# In[3]:


x=pd.DataFrame(data['YearsExperience'])


# In[4]:


y=pd.DataFrame(data['Salary'])


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)


# In[7]:


print(x_train.shape) # checking out the shape of the training dataset  


# In[8]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[9]:


print(reg.intercept_)


# In[10]:


y_pred=reg.predict(x_test)
y_pred=pd.DataFrame(y_pred,columns=['predicted'])


# In[11]:


y_pred.head


# In[12]:


y_test


# In[14]:


from sklearn.metrics import r2_score
print("coefficent of Determination or R2 score:{}".format(r2_score(y_test,y_pred)))


# In[ ]:





