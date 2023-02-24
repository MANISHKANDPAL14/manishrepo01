#!/usr/bin/env python
# coding: utf-8

# ## TITANIC CASE STUDY (Working of logistic Regression)
# 

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[42]:


train = pd.read_csv(r"C:\Users\Abhinash\Downloads\titanic_train.csv")
# The r is called raw strings and is placed before the filename to prevent the characters in the filename string to be treated
# as special characters


# In[44]:


train.head(2)


# ### data Dictionary
# - Passenger ID: unique passenger ID
# - Pclass: In titanic there were 3 class : 1st(Premium business) , 2nd(busniness) , 3rd(economy) class passenger
# - Name: passenger name
# - Gender : Male | Female
# - Age : passenger age
# - Sibsp : # of siblings / spouses aboard the Titanic
# - Prach: no of childern
# - ticket : ticket number
# - fare : ticket fare
# - cabin : the passenger assign to which compartment
# - embarked: port from where the person board the ship ( C = Cherbourg, Q = Queenstown, S = Southampton)
# - boat : life raft number
# - body : No info on this columns
# - home.dest : final destination
# - survival: dependent variable --> person survived the disaster or not 0(not survive) | 1 (survived)
# 

# In[1]:


# data cleaning 
# data preprocessing
# feature selection
# today: donwload the data and try to clean and do preprocessing, feature selection


# In[ ]:


# create ML model(regression)
# how to create the Sen & spec Vs threshold
# how to find best threshold
# create ROC curve and understanding
# Precison recall and F1 score


# In[ ]:


# how to find best parameter-> hyper paramter tunning


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# The r is called raw strings and is placed before the filename to prevent the characters in the filename string to be treated
# as special characters
df = pd.read_csv(r"C:\Users\Abhinash\Desktop\DSP TIME 1.30-3.30PM DATE 13 NOV 2021\titanic datasets\titanic_train.csv")
df.head()


# In[4]:


# precentage of missing data in all columns
per = df.isnull().sum() / len(df)
per


# In[5]:


df.columns


# In[6]:


# unnecessary columns removed and only important columns are taken
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch',
       'fare', 'embarked', 'survived']]
df.head()


# In[7]:


#checking numerical and categorical columns
df.dtypes


# In[9]:


#filling missing data in age column
df["age"].fillna(df["age"].median(),inplace=True)


# In[10]:


df.isnull().sum()


# In[11]:


df["fare"].fillna(df["fare"].median(),inplace=True)


# In[12]:


df["embarked"].value_counts()


# In[13]:


df["embarked"].fillna("S",inplace=True) 


# In[14]:


df.isnull().sum()


# In[15]:


df.head()


# In[16]:


# text---> numeric(LE)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[17]:


df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"])


# In[18]:


df.head()


# In[19]:


Y = df["survived"]
X = df.drop(["survived"],axis=1)


# In[20]:


X.head()


# In[21]:


# call sklearn.linear_model
from sklearn.linear_model import LogisticRegression


# In[22]:


model = LogisticRegression()


# In[23]:


# call train test split which will divide my overall data into two parts
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)


# In[25]:


# train the model with the default value
get_ipython().run_line_magic('pinfo', 'LogisticRegression')
LogisticRegression(
    penalty='l2',
    *,
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='lbfgs',
    max_iter=100,
    multi_class='auto',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
)


# In[83]:


# lets train the model over all data 
#model.fit(X,Y)
model.fit(x_train,y_train)


# In[84]:


# testing we will use the testing (train test split)


# In[85]:


# by default the value of threshold is 0.50
pred_default = model.predict(x_test)
pred_default


# In[86]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred_default))


# In[87]:


pred_prob = model.predict_proba(x_test)
pred_prob


# In[88]:


df2 = pd.DataFrame(data=pred_prob,columns=["Not Survived","Survived"])
df2.head()


# In[89]:


# we will use the concept of list comprehension to generate a threshold set
th=[i/100 for i in range(1,101)]
print(th)


# In[90]:


# we will create multiple columns for each threshold
for i in th:
    df2["{}-Threshold".format(i)] = np.where(df2["Survived"]> i ,1 ,0)    


# In[91]:


df2.head()


# In[92]:


df3 = df2.drop(["Not Survived","Survived"],axis=1)
df3.head()


# In[93]:


from sklearn.metrics import accuracy_score


# In[94]:


# for each trheshold we can calulate accuracy for model
acc=[]
for i in df3.columns:
    acc.append(accuracy_score(y_test,df3[i]))


# In[95]:


df4 = pd.DataFrame({"Threshold":th,"Accuracy":acc})
df4.head()


# In[96]:


# what will be the threshold where this model accuray is high
highest_th_acc = df4.sort_values(by="Accuracy",ascending=False)
highest_th_acc.head(1)


# In[97]:


# we will create a visulaization using threshold and your acc
plt.figure(figsize=(6,6))
plt.plot(df4["Threshold"],df4["Accuracy"])
plt.show()                                                                                                                                                          


# ### how to say that model is good at 0.59 threshold
# - look into the concept of Specficity and Sensitvity to look for the case
# - I cannot say directly looking towards my accuracy i have to make a model good in order to response for Neg|Pos case equally

# ![R.gif](attachment:R.gif)

# In[98]:


# based on default
# by default the value of threshold is 0.50
from sklearn.metrics import confusion_matrix
pred_default = model.predict(x_test)
confusion_matrix(y_test,pred_default)


# In[111]:


#[[TN(110),  FP(19)],
#[FN(25),  TP(59)]]
# sen : TP / (Tp+Fn)
# spec: TN / (TN+FP)


# In[100]:


cm=confusion_matrix(y_test,pred_default)


# In[101]:


cm


# In[102]:


TP = cm[1:,1:][0][0]
print(TP)
TN = cm[0][0]
print(TN)
dem_sen = cm[1:,0:].flatten().sum()
print(dem_sen)
dem_spec = cm[0:1,0:].flatten().sum()
print(dem_spec)


# ![R.gif](attachment:R.gif)

# In[103]:


df3.head()


# In[115]:


# for each trheshold we can calulate accuracy for model
def final_snes_spec_df(actual,df_th):
    '''this function return a data frame which contain following matrices'''
    acc = []
    sens = []
    sepcf = []
    for i in df_th.columns:
        cm = confusion_matrix(y_test,df_th[i])
        TP = cm[1:,1:][0][0]
        TN = cm[0][0]
        dem_sen = cm[1:,0:].flatten().sum()
        dem_spec = cm[0:1,0:].flatten().sum()
        SEN = TP/dem_sen
        SPEC = TN/dem_spec
        sens.append(SEN)
        sepcf.append(SPEC)
        acc.append(accuracy_score(y_test,df3[i]))
    final_ok = pd.DataFrame({"threshold":th,
                     "Accuracy":acc,
                     "sensitivity":sens,
                     "Specificity":sepcf})
    return final_ok


# In[116]:


get_ipython().run_line_magic('pinfo', 'final_snes_spec_df')


# In[113]:


final=final_snes_spec_df(y_test,df3)


# In[114]:


final.head()


# In[106]:


# SENSITIVITY & SPECIFICITY  Vs Threshold(cutoff)
plt.figure(figsize=(6,6))
plt.plot(final["threshold"],final["sensitivity"],color ="red",label="Sensitivity")
plt.plot(final["threshold"],final["Specificity"],color ="blue",label="Specificity")
plt.plot(final["threshold"],final["Accuracy"],color ="black",label="Accuracy")
plt.xlabel("Cutoff")
plt.ylabel("SENS & SPEC")
plt.legend()
plt.show()


# In[ ]:





# TN(110),  FP(19)<br>
# FN(25),  TP(59)<br>
# 
# we are not concern about TP|TN <br>
# **we are concern of FP & FN becasue if my model say that the person is survived but actually the person died ----> FP** <br>
# This point is also not a big concern when he model says that the person will not survived but he survived

# ## ROC Curve
# - ROC : Reciver Operating Characteristcs
# - Roc is plot between Sensitivity Vs (1-Specificity)

# In[107]:


final.head()


# In[108]:


final["1-Specificity"] = 1-final["Specificity"]


# In[109]:


final.head()


# In[110]:


# SENSITIVITY & SPECIFICITY  Vs Threshold(cutoff)
plt.figure(figsize=(6,6))
plt.plot(final["1-Specificity"],final["sensitivity"])
plt.ylabel("SENS")
plt.xlabel("1-Specificity")
plt.title("ROC CURVE")
plt.show()


# ![maxresdefault.jpg](attachment:maxresdefault.jpg)
