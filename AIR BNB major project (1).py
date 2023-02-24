#!/usr/bin/env python
# coding: utf-8

# ## AIRBNB NEWYORK REGRESSION MODEL MAJOR PROJECT
# - data link : https://drive.google.com/file/d/1MCIPN-8orBAJ88ucWiLfUWBe5Dj4bw5G/view?usp=sharing
# - data loading 
# - data cleaning
# - feature selection
# - feature split - dep & indp
# - data split - train/test
# - Model training
# - model testing
# - Model Evalution

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# loading data
df = pd.read_csv(r"C:\Users\HP\Downloads\Air Bnb.csv")
df.head(2)


# In[3]:


df.describe()


# ## data cleaning and preprocessing
# - null values

# In[4]:


df.isnull().sum()


# In[110]:


#df.isnull().sum().plot.barh()


# In[6]:


cleaned_data = df.drop(['id','description','first_review','last_review','host_since','host_has_profile_pic','name',
                        'thumbnail_url','zipcode','neighbourhood'],axis=1)
cleaned_data.columns


# In[7]:


# Those columns are needed to be processed
"""
 5   bathrooms               73911 non-null  float64
 10  host_identity_verified  73923 non-null  object 
 11  host_response_rate      55812 non-null  object 
 15  neighbourhood           67239 non-null  object 
 17  review_scores_rating    57389 non-null  float64
 18  bedrooms                74020 non-null  float64
 19  beds                    73980 non-null  float64
"""
cleaned_data.info()


# In[8]:


cleaned_data.isnull().sum()


# In[9]:


#cleaned_data.host_identity_verified = cleaned_data.host_identity_verified.fillna(cleaned_data.host_identity_verified.mode())


# In[10]:


# Numeric values fill with median of each column of it
cleaned_data.bathrooms = cleaned_data.bathrooms.fillna(int(cleaned_data.bathrooms.median()))
cleaned_data.bedrooms = cleaned_data.bedrooms.fillna(int(cleaned_data.bedrooms.median()))
cleaned_data.beds = cleaned_data.beds.fillna(int(cleaned_data.beds.median()))


# In[11]:


# Fixing host response rate and change its value to the correct data type
cleaned_data.review_scores_rating=cleaned_data.review_scores_rating.fillna(cleaned_data.review_scores_rating.median())

#The apply() method allows you to apply a function along one of the axis of the DataFrame, default 0, which is the index (row)
#axis.
cleaned_data.host_response_rate=cleaned_data.host_response_rate.apply(lambda x:int(x[:len(x)-1])/100 if isinstance(x,str) else x)
cleaned_data.host_response_rate=cleaned_data.host_response_rate.fillna(cleaned_data.host_response_rate.mean())


# In[12]:


cleaned_data.isnull().sum()


# In[13]:


cleaned_data.head(2)


# In[14]:


cleaned_data.host_identity_verified


# In[15]:


cleaned_data.host_identity_verified = np.where(cleaned_data.host_identity_verified =="t",1,0)


# In[16]:


cleaned_data.isnull().sum()


# In[17]:


cleaned_data.host_identity_verified.value_counts()


# In[18]:


cleaned_data = cleaned_data.dropna()


# In[19]:


cleaned_data.shape


# In[20]:


# Changing all boolean objects to 0/1
cleaned_data.host_identity_verified=cleaned_data.host_identity_verified.apply(lambda x: 1 if x=='t' else 0)


# In[21]:


cleaned_data.head(2)


# In[22]:


cleaned_data.bed_type.value_counts()


# In[23]:


cleaned_data.amenities[0]


# In[24]:


cleaned_data.instant_bookable=cleaned_data.instant_bookable.apply(lambda x: True if x=='t' else False)


# In[25]:


# Factorization of categorical columns
"""
 1   property_type           74111 non-null  object # 35 <-----------------
 2   room_type               74111 non-null  object # 3  - 
 6   bed_type                74111 non-null  object # 5
 7   cancellation_policy     74111 non-null  object # 5
 9   city                    74111 non-null  object # 6
"""


# In[26]:


cleaned_data.room_type.unique()


# In[27]:


cleaned_data.room_type=np.where(cleaned_data.room_type=="Entire home/apt",1,
                                 np.where(cleaned_data.room_type=="Private room",2,3))
#cleaned_data.room_type=cleaned_data.room_type.apply(lambda x: 3 if x=='Entire home/apt' else 2 if x=='Private room' else 1)


# In[28]:


cleaned_data.bed_type.unique()


# In[29]:


cleaned_data.bed_type = cleaned_data.bed_type.apply(lambda x: 2 if x=='Real Bed' else 1)


# In[30]:


cleaned_data.head(2)


# In[31]:


cleaned_data.cancellation_policy.unique()


# In[32]:


#cleaned_data.cancellation_policy = cleaned_data.cancellation_policy.apply(lambda x: 3 if x=='super_strict_60' else 2 
#                                    if x=='super_strict_30' else 1)
cleaned_data.cancellation_policy = np.where(cleaned_data.cancellation_policy=="moderate",2,
                                           np.where(cleaned_data.cancellation_policy=="flexible",3,1))


# In[33]:


cleaned_data.head(3)


# In[34]:


cleaned_data['city']


# In[35]:


city_dum=pd.get_dummies(cleaned_data['city'],prefix='city')
city_dum


# In[36]:


cleaned_data=pd.concat([cleaned_data, pd.get_dummies(cleaned_data['city'], prefix='city')],axis=1)


# In[37]:


cleaned_data.head(2)


# In[38]:


cleaned_data = cleaned_data.drop(['city'],axis=1)


# In[39]:


cleaned_data = cleaned_data.drop(['latitude','longitude'],axis=1)


# In[40]:


cleaned_data.head(2)


# In[41]:


v1 = cleaned_data[["amenities"]]
v1


# In[42]:


v1["len amenities"]=[len(i) for i in v1.amenities]
v1.head()


# In[43]:


amenities_col = []
for s in v1.amenities:
  s = s.replace('{','')
  s = s.replace('}','')
  s = s.replace('"','')
  s = s.split(',')
  val = max(len(s)-1,0)
  print(val)
  amenities_col.append(max(len(s)-1,0))
print(amenities_col)


# In[44]:


#Composite type attr.
amenities_col = []
amenities_map = {}
for s in cleaned_data.amenities:
  s = s.replace('{','')
  s = s.replace('}','')
  s = s.replace('"','')
  s = s.split(',')
  amenities_col.append(max(len(s)-1,0))
  for k in s:
    if amenities_map.get(k) != None:
      amenities_map[k] +=1 
    else:
      amenities_map[k]=1


# In[45]:


cleaned_data['amenities_count'] = pd.Series(amenities_col)
cleaned_data.head(3)


# In[46]:


cleaned_data = cleaned_data.drop(['amenities'], axis=1)
cleaned_data.review_scores_rating = cleaned_data.review_scores_rating/100


# In[47]:


cleaned_data.head(3)


# In[48]:


cleaned_data.property_type.unique()


# In[49]:


import plotly.figure_factory as ff
corrs=cleaned_data.corr()
# plt.figure(figsize=(10,10))
# figure=ff.create_annotated_heatmap(
#     z=corrs.values,
#     x=list(corrs.columns),
#     y=list(corrs.index),
#     annotation_text=corrs.round(2).values,
#     showscale=True)
# figure.show()


# In[50]:


import seaborn as sns
sns.boxplot(cleaned_data.bedrooms)


# In[51]:


cleaned_data.bedrooms.describe()


# In[52]:


print(np.quantile(cleaned_data.bedrooms,[i/100 for i in range(1,101)]))


# In[53]:


## Removing correlated features and unneeded ones
cleaned_data = cleaned_data.drop(['beds','bathrooms','bedrooms','instant_bookable','number_of_reviews','host_response_rate']
                                 ,axis=1)


# In[54]:


#cleaned_data=cleaned_data.drop(['host_response_rate',],axis=1)


# In[55]:


cleaned_data.pop("host_identity_verified")


# In[56]:


cleaned_data.head(2)


# In[57]:


cleaned_data["property_type"].value_counts()


# In[58]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cleaned_data["property_type"]=le.fit_transform(cleaned_data["property_type"])


# In[59]:


cleaned_data.cleaning_fee=np.where(cleaned_data.cleaning_fee==True,1,0)
cleaned_data.head(2)


# In[60]:


corrs=cleaned_data.corr()


# In[61]:


corrs.to_csv("correlation.csv")


# In[62]:


# in machine learning : technique called Random forest/decison tree


# In[63]:


# Let's load the packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt


# In[64]:


Y = cleaned_data.log_price
X = cleaned_data.drop(["log_price"],axis=1)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X,Y)


# In[65]:


rf.feature_importances_


# In[66]:


len(rf.feature_importances_)


# In[67]:


plt.barh(X.columns,rf.feature_importances_)


# #OLS or Ordinary Least Squares is a method used in Linear Regression for estimating the unknown parameters by creating a model
# #which will minimize the sum of the squared errors between the observed data and the predicted one

# #formula='log_price~property_type+room_type+accommodates+bed_type+cancellation_policy+cleaning_fee+review_scores_rating+city_Boston+city_Chicago+city_DC+city_LA+city_NYC+city_SF+amenities_count'

# In[68]:


#Statsmodels formula
from statsmodels.formula.api import ols
formula = 'log_price~property_type+room_type+accommodates+bed_type+cancellation_policy+cleaning_fee+review_scores_rating +city_Boston+city_Chicago+city_DC+city_LA+city_NYC+city_SF+amenities_count'
model=ols(formula=formula,data=cleaned_data).fit()
model.summary()


# In[ ]:


#Statsmodels formula
cleaned_data.pop("bed_type")
from statsmodels.formula.api import ols
formula ='log_price~property_type+room_type+accommodates+cancellation_policy+cleaning_fee+review_scores_rating+city_Boston+city_Chicago+city_DC+city_LA+city_NYC+city_SF+amenities_count'
model = ols(formula=formula, data=cleaned_data).fit()
model.summary()


# In[108]:


import seaborn as sns
plt.figure(figsize=(12,10))
corrs=cleaned_data.corr()
sns.heatmap(corrs,annot=True)


# In[71]:


X = cleaned_data.drop(["log_price"],axis=1)
Y = cleaned_data.log_price


# In[72]:


cleaned_data.shape


# In[73]:


# Spliting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)


# In[74]:


x_train.shape


# In[75]:


x_test.shape


# In[76]:


### data normalization
## data split
## model training
# model testing


# In[77]:


get_ipython().system('pip install plotly')


# Scaling the data
# 
# Q.Why we use Fit_transform () on training data but transform () on the test data?
# 
# fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of 
# that data. Here, the model built by us will learn the mean and variance of the features of the training set. These learned
# parameters are then used to scale our test data. x_test is data that our model does not have knowledge of. 
# This is why we need to transform it based on what our model knows, and hence we transform x_test using the values learnt from
# x_train during the . fit method.

# In[78]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
X_trn = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


# In[79]:


from sklearn.linear_model import LinearRegression


# In[80]:


Model = LinearRegression()


# In[81]:


# X is inputs and Y is output.Xtrain  and Ytrain are the data that from which the model is to be trained.
Model.fit(X_trn,y_train)


# In[82]:


#input test data ie xtest is given to a learned model to predict the predicted values
pred = Model.predict(X_test)


# In[83]:


pred


# In[84]:


from sklearn.metrics import r2_score,mean_squared_error


# In[85]:


#now estimating the r2 score with the help of predicted output and the actual output
r2_score(y_test,pred)


# In[86]:


#now estimating the MSE with the help of predicted output and the actual output
mean_squared_error(y_test,pred)


# In[87]:


df1 = pd.DataFrame({"actual_data":y_test,"predicted_data":pred})


# In[88]:


import seaborn as sns
sns.lmplot("actual_data","predicted_data",data=df1)


# In[89]:


get_ipython().run_line_magic('pinfo', 'sns.lmplot')


# ## How to test weather the model is good or bad
# - Bias & Variance Tradeoff
#    - Good model
#    - Bad model
#    - Underfit model
#    - Overfit Model
#    
# # Developer Validation Testing
# - We divide the data into 3 parts
#  - Training data - 70-80%
#  - testing data - 30-20%
#  - validation data - unseen data - sample data from orignal data -10-20%

# In[90]:


cleaned_data.head()


# In[91]:


cleaned_data.shape[0]*.10


# In[109]:


# validation or random data from overall data
val_data = cleaned_data.sample(n=7412)
val_data.head(4)


# In[93]:


# Spliting the data
X = cleaned_data.drop(["log_price"],axis=1)
Y = cleaned_data["log_price"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.25, random_state=42)


# In[94]:


x_train.shape


# In[95]:


x_test.shape


# In[96]:


from sklearn.linear_model import LinearRegression
Model = LinearRegression()
Model.fit(x_train,y_train)


# In[97]:


pred = Model.predict(x_train)
from sklearn.metrics import r2_score


# In[98]:


r2_score(y_train,pred)


# In[99]:


train_acc = Model.score(x_train,y_train)


# In[100]:


test_acc = Model.score(x_test,y_test)


# In[101]:


X_val = val_data.drop(["log_price"],axis=1)
Y_val = val_data["log_price"]


# In[102]:


Val_acc = Model.score(X_val,Y_val)


# In[103]:


# these score which we got a  model - accuracy of models
print("Model Acc during Train",train_acc)
print("Model Acc during test",test_acc)
print("Model Acc during val",Val_acc)


# In[104]:


# Variance and Bias
#Bias:mean the value of all given accuracy
#Variance:how much the value is from the mean position


# In[105]:


Bias=np.mean([0.614,0.316,0.416])
Bias


# In[106]:


Bias = np.mean([train_acc,test_acc,Val_acc])
Bias


# In[107]:


var = np.var([train_acc,test_acc,Val_acc])
var


# - When to say model is good - condition : When model is having Low Bias and Low Variance
# - When we say good model which means during training model was able to capture hidden pattern correctly and 
# - during testing model was able to capture the unseeen data correctly

# - when the model is having High Bias and High Variance 
# - During training and testing model is not able to find good pattern or it got confused

# - Underfit model bias is high and variance is low
# - During Training model becomes bias towards one class or one result and not able to understand other features.

# - OverFit Model : when the Bias is low and Variance is high
# - during training you model get a very different data or there was high variance due to which it leanr a lot of thing and got confused and during testing it not able to predict correclty
