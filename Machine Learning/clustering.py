#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load data
df_customer = pd.read_csv("E:\Certifications\Machine Learning & Data Science\Kalbe\data\Customer.csv", delimiter = ';')
df_transaction = pd.read_csv("E:\Certifications\Machine Learning & Data Science\Kalbe\data\Transaction.csv", delimiter = ';')


# In[3]:


# Check transaction dataframe info
df_transaction.info()


# Setelah melihat info dari data transaksi, tidak ditemukan adanya data kosong. namun, pada kolom date terdeteksi sebagai tipe data objek.

# In[4]:


# convert Date to datetime
df_transaction['Date'] = pd.to_datetime(df_transaction['Date'], format='%d/%m/%Y')


# In[5]:


# check transaction dataframe info after convert date
df_transaction.info()


# In[6]:


df_customer.head()


# In[7]:


# check customer dataframe info
df_customer.info()


# In[8]:


df_customer[df_customer['Marital Status'].isnull()]


# In[9]:


# fill missing values on df_customer
df_customer.fillna(method='ffill', inplace=True)

# convert categorical data to numerical
df_customer['Marital Status'] = df_customer['Marital Status'].apply(lambda x: 1 if x == 'Married' else 0)

# convert Income to float
df_customer['Income'] = df_customer['Income'].apply(lambda x: x.replace(',', '.')).astype(float)


# In[10]:


df_customer.head()


# In[11]:


df_customer.info()


# In[12]:


# merge df_transaction and df_customer
merged_df = pd.merge(df_transaction, df_customer, on='CustomerID', how='left')
merged_df.head()


# In[13]:


merged_df.info()


# In[42]:


# aggregate data
agg = {
    'TransactionID': 'count',
    'Qty': 'sum',
    'Age' : 'first'
}
cluster_df = merged_df.groupby('CustomerID').aggregate(agg).reset_index()
cluster_df.info()


# In[43]:


# scale data into same range
scaler = StandardScaler()
scaled_df = scaler.fit_transform(cluster_df[['TransactionID', 'Qty', 'Age']])
scaled_df = pd.DataFrame(scaled_df, columns=['TransactionID', 'Qty', 'Age'])
scaled_df.head()


# In[44]:


# finding optimal number of clusters
inertia = []
max_clusters = 11
for n_cluster in range(1, max_clusters):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=n_cluster)
    kmeans.fit(cluster_df.drop('CustomerID', axis=1))
    inertia.append(kmeans.inertia_)


# In[45]:


plt.figure(figsize=(10,8))
plt.plot(np.arange(1, max_clusters), inertia, marker='o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.xticks(np.arange(1, max_clusters))
plt.show()


# In[53]:


# create cluster
n_cluster = 4
kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=n_cluster)
kmeans.fit(cluster_df.drop('CustomerID', axis=1))
cluster_df['Cluster'] = kmeans.labels_


# In[66]:


# plot cluster
cluster_df.plot(kind='scatter', x='Qty', y='Age', c='Cluster', cmap='viridis', figsize=(6,4), legend=True)


# In[62]:


cluster_df.groupby(['Cluster']).agg({
    'CustomerID' : 'count',
    'Qty' : 'mean',
    'Age' : 'first',
    'TransactionID' : 'count'
}).rename(columns={
    'CustomerID' : 'Customer_Count',
    "TransactionID" : 'Transaction_Count'
})

