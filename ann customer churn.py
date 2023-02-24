#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()


# In[3]:


df.drop('customerID',axis='columns',inplace=True)
df.dtypes


# In[4]:


df.TotalCharges.values


# In[5]:


df.MonthlyCharges.values


# In[6]:


pd.to_numeric(df.TotalCharges,errors='coerce')


# In[7]:


pd.to_numeric(df.TotalCharges,errors='coerce').isnull().sum()


# In[8]:


df.isnull().sum()


# In[9]:


df.shape


# In[10]:


df.iloc[448]


# In[11]:


df.iloc[448]['TotalCharges']


# In[12]:


df1=df[df.TotalCharges!=' ']
df1.shape


# In[13]:


df1.dtypes


# In[14]:


df1.TotalCharges=pd.to_numeric(df1.TotalCharges)


# In[15]:


df1.TotalCharges.dtypes    #converted to float


# In[16]:


tenure_churn_no=df1[df1.Churn=='No'].tenure
tenure_churn_yes=df1[df1.Churn=='Yes'].tenure
plt.hist([tenure_churn_no,tenure_churn_yes],color=['green','red'],label=['churn=no','churn=yes'])
plt.legend()
plt.xlabel('no of customers')
plt.ylabel('tenure')
plt.title('customer churn prediction visualization')


# In[17]:


df1


# In[18]:


mc_churn_no=df1[df1.Churn=='No'].MonthlyCharges
mc_churn_yes=df1[df1.Churn=='Yes'].MonthlyCharges
plt.hist([mc_churn_no,mc_churn_yes],color=['green','red'],label=['churn=no','churn=yes'])
plt.legend()
plt.xlabel('no of customers')
plt.ylabel('MonthlyCharges')
plt.title('customer churn prediction visualization')


# In[19]:


# for column in df:
#     print(df[column].unique())
for column in df:
    if df[column].dtypes=='object':
        print(f'{column}:{df[column].unique()}')


# In[20]:


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column}:{df[column].unique()}')


# In[21]:


df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)


# In[22]:


print_unique_col_values(df1)


# In[23]:


yes_no_columns=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes':1,'No':0},inplace=True)


# In[24]:


print_unique_col_values(df1)


# In[25]:


df1['gender'].replace({'Female':1,'Male':0},inplace=True)
df1['gender'].unique()


# In[26]:


df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])
df2.columns


# In[27]:


df2.sample(4)


# In[28]:


df2.dtypes


# In[29]:


cols_to_scale=['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df2[cols_to_scale]=scaler.fit_transform(df2[cols_to_scale])
df2


# In[30]:


for col in df2:
    print(f'{col}:{df2[col].unique()}')


# In[31]:


y= df2.Churn
x= df2.drop(["Churn"],axis=1)


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


# In[33]:


x_train.shape


# In[34]:


x_test.shape


# In[35]:


len(x_train.columns)


# In[39]:


import tensorflow as tf
from tensorflow import keras


model=keras.Sequential([
    keras.layers.Dense(20,input_shape=(26,),activation='relu'),
    keras.layers.Dense(1,input_shape=(26,),activation='sigmoid'),
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(x_train,y_train,epochs=100)


# In[41]:


model.evaluate(x_test,y_test)


# In[42]:


yp=model.predict(x_test)
yp[:5]


# In[45]:


y_test[:10]


# In[43]:


y_pred=[]
for element in yp:
    if element>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[46]:


y_pred[:10]


# In[49]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[50]:


print(confusion_matrix(y_test,y_pred))


# In[53]:


import seaborn as sns
cm=tf.math.confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicter')
plt.ylabel('truth')


# In[59]:


accuracy=(906+194)/(906+194+127+180)
round(accuracy,2)


# In[66]:


#precision for 0 class
precision=(906)/(906+180)
precision


# In[67]:


#precision for 1 class
precision1=(194)/(194+127)
precision1


# In[69]:


recall0=(906)/(906+127)
round(recall0,2)


# In[70]:


recall1=(194)/(194+180)
round(recall1,2)

