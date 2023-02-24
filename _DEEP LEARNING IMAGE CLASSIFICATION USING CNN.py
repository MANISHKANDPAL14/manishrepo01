#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,models,layers


# In[3]:


(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data()


# In[4]:


X_train


# In[5]:


y_train


# In[6]:


y_train=y_train.reshape(-1,)
y_train[:5]


# In[7]:


X_test


# In[8]:


y_test


# In[9]:


X_train.shape


# In[10]:


X_test.shape


# In[11]:


y_train.shape


# In[12]:


y_train.ndim


# In[13]:


y_test.shape


# In[14]:


plt.imshow(X_train[0])


# In[15]:


plt.imshow(X_train[1])


# In[16]:


classes=["airplane","truck","frog","cat","bird","deer","dog","ship","horse","automobile"]
classes.sort()
classes


# In[17]:


y_train[:5]


# In[18]:


def plot_sample(X,y,index):
    plt.imshow(X[index]) 
    plt.xlabel(classes[y[index]])


# In[19]:


plot_sample(X_train,y_train,5)


# In[20]:


plot_sample(X_train,y_train,2)


# In[21]:


X_train=X_train/255
X_test=X_test/255


# In[22]:


X_train[0]


# In[23]:


X_test[0]


# In[24]:


ann=models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(3000,activation='relu'),
    layers.Dense(1000,activation='relu'),
    layers.Dense(10,activation='sigmoid')
])

ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train,y_train,epochs=5)


# In[34]:


ann.evaluate(X_train,y_train)


# In[37]:


from sklearn.metrics import confusion_matrix ,classification_report
y_pred=ann.predict(X_test)
y_pred_classes=[np.argmax(element) for element in y_pred]
print(classification_report(y_test,y_pred_classes))


# In[38]:


confusion_matrix(y_test,y_pred_classes)


# In[30]:


X_train.shape


# In[31]:


plt.imshow(X_train[1])


# In[50]:


cnn=models.Sequential([
    
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters=64,kernel_size=(5,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])


# In[51]:


cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


# In[52]:


cnn.fit(X_train,y_train,epochs=5)


# In[57]:


y_test=y_test.reshape(-1,)
y_test[:5]


# In[58]:


plot_sample(X_test,y_test,1)


# In[59]:


y_pred=cnn.predict(X_test)
y_pred[:5]


# In[60]:


np.argmax([1,12,34,4]) #it returns the index of max value


# In[61]:


np.argmax(y_pred[0])


# In[65]:


y_classes=[np.argmax(element) for element in y_pred]
y_classes[:10]


# In[66]:


y_test[:10]


# In[67]:


plot_sample(X_test,y_test,0)


# In[70]:


classes[y_classes[0]]


# In[68]:


plot_sample(X_test,y_test,1)


# In[69]:


classes[y_classes[1]]

