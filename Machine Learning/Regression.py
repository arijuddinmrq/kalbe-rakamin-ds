#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot


# In[3]:


df_customer = pd.read_csv("E:\Certifications\Machine Learning & Data Science\Kalbe\data\Customer.csv", delimiter = ';')
df_product = pd.read_csv("E:\Certifications\Machine Learning & Data Science\Kalbe\data\Product.csv", delimiter = ';')
df_store = pd.read_csv("E:\Certifications\Machine Learning & Data Science\Kalbe\data\Store.csv", delimiter = ';')
df_transaction = pd.read_csv("E:\Certifications\Machine Learning & Data Science\Kalbe\data\Transaction.csv", delimiter = ';')


# ## Data Cleansing

# ### df_customer

# In[4]:


df_customer.info()


# In[5]:


df_customer.isnull().sum()


# In[6]:


df_customer.fillna(method='ffill', inplace=True)


# In[7]:


df_customer.duplicated().sum()


# In[8]:


df_customer['Income'] = df_customer['Income'].apply(lambda x: x.replace(',', '.')).astype(float)


# In[9]:


df_product


# In[10]:


df_store


# In[11]:


df_transaction.head()


# In[12]:


df_transaction.info()


# In[13]:


df_transaction['Date'] = pd.to_datetime(df_transaction['Date'])


# ## Merge Data

# In[14]:


df_merge = pd.merge(df_transaction, df_customer, on=['CustomerID'])
df_merge = pd.merge(df_merge, df_product, on=['ProductID'])
df_merge = pd.merge(df_merge, df_store, on=['StoreID'])
df_merge.head()


# In[15]:


df_merge = df_merge.drop(columns=['Price_y'])


# In[16]:


df_merge = df_merge.rename(columns={'Price_x' : 'Price'})


# In[17]:


df_merge.info()


# ## Regression Model (Time Series)

# In[18]:


df_reg = df_merge.groupby(['Date']).agg({
    'Qty' : 'sum'
})


# In[19]:


df_reg


# In[20]:


# plot qty sales in a year
df_reg.plot(figsize=(12,8), title='Daily Sales', xlabel='Date', ylabel='Total Qty', legend=False)


# In[21]:


x_train = df_reg[:int(0.8*(len(df_reg)))].reset_index()
x_test = df_reg[int(0.8*(len(df_reg))):].reset_index()


# In[22]:


x_train


# In[23]:


x_test


# In[25]:


plt.figure(figsize = (20,5))
sns.lineplot(data=x_train, x=x_train['Date'], y=x_train['Qty']);
sns.lineplot(data=x_test, x=x_test['Date'], y=x_test['Qty']);


# In[26]:


def rmse(y_actual,y_pred):
    print(f'RMSE value {(mean_squared_error(y_actual, y_pred)**0.5)}')
def eval(y_actual, y_pred):
    rmse(y_actual,y_pred)
    print(f'MAE value {mean_absolute_error(y_actual, y_pred)}')


# In[27]:


train = x_train.set_index('Date')
test = x_test.set_index('Date')

y = train['Qty']

ARIMAmodel = ARIMA(y,order = (4, 2, 1))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(test))

y_pred_df = y_pred.conf_int()
y_pred_df['predictions'] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df['predictions']
eval(test['Qty'], y_pred_out)

plt.figure(figsize=(20,5))
plt.plot(train['Qty'])
plt.plot(test['Qty'])
plt.plot(y_pred_out, color = 'black', label = 'ARIMA Predict')
plt.legend()


# In[28]:


y_pred_df


# In[29]:


df_qty =test['Qty']
df_qty_pred = y_pred_df['predictions']
df_qty_pred.shape


# In[30]:


myDict = {
    'Data Real' : df_qty, 'Prediksi' : df_qty_pred
}


# In[31]:


df_predict = pd.DataFrame(myDict)
df_predict

