#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import datetime
import pandas_datareader as pdr


# In[21]:


df=pd.read_csv('HistoricalPrices.csv',parse_dates=['Date'])
# index_col='DATE'
# df.drop(['Open','High','Low'],axis=1,inplace=True)

# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df['Date']=pd.to_datetime(df['Date'])


# In[22]:



df=df.sort_values(by='Date',ascending=True)


# In[23]:


df.drop([' Open', ' High', ' Low'],axis=1,inplace=True)
df.reset_index(inplace=True,drop=True)


# In[24]:


df


# In[25]:


list(df.columns)


# In[26]:


def MStd(price,days):
    ma=price.rolling(days).std()
    ma.dropna(inplace=True)
    return ma

def MAvg(price,days):
    ma=price.rolling(days).mean()
    ma.dropna(inplace=True)
    return ma

def MMax(price,days):
    ma=price.rolling(days).max()
    ma.dropna(inplace=True)
    return ma

def MMin(price,days):
    ma=price.rolling(days).min()
    ma.dropna(inplace=True)
    return ma


# In[27]:


a=MStd(df,30)
a=a.rename({' Close':'D&J_Close30d_Std.'},axis=1)
a


# In[28]:


list(a.columns)


# In[29]:


b=MAvg(df,30)
b=b.rename({' Close':'D&J_Close30d_Avg.'},axis=1)
b


# In[30]:


c=MMax(df,30)
c=c.rename({' Close':'D&J_Close30d_Max'},axis=1)
c


# In[31]:


d=MMin(df,30)
d=d.rename({' Close':'D&J_Close30d_Min'},axis=1)
d


# In[32]:


df=pd.merge(df,a,how='inner',left_index=True,right_index=True)
df=pd.merge(df,b,how='inner',left_index=True,right_index=True)
df=pd.merge(df,c,how='inner',left_index=True,right_index=True)
df=pd.merge(df,d,how='inner',left_index=True,right_index=True)


# In[33]:


df.head()


# In[34]:


df[['D&J_Close30d_Avg.','D&J_Close30d_Max','D&J_Close30d_Min']].plot(figsize=(12,5))


# In[35]:


# Filter the first date of each month 
df=df.resample('M',on='Date').first().dropna().reset_index(drop=True)
df.head()
list(df.columns)


# In[36]:


df.drop([' Close'],axis=1,inplace=True)
df


# In[38]:


pd.DataFrame.to_csv(df,'Monthly_new.csv')


# In[ ]:





# In[ ]:




