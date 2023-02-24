#!/usr/bin/env python
# coding: utf-8

# ## Supervised Learning - Classification
# 
# It is a type of Supervised Learning that learns from the labelled data and predict the category the data belongs to like spam detection, Loan Defaulter etc.

# ### Naive Bayes

# It is easy and simple classification technique. If we have a large dataset and we want to build a model as soon as possible then we have Naive Bayes model which is quite fast learner than the other classification models.
# 
# It assumes the features independency upon the other features wheather the feature depends or rely on  the other features, it assumes the contribution of each feature independently that is why it is called **Naive**.
# 
# It uses Bayes theorem to calculate the probability of an event to be occured based on the past event that already has been occured.
# 
# **P(c|X) = P(X|c)*P(c)/P(X)**
# 
# P(c|X) = Posterior Probability of event c occuring when event X has occured.
# P(X|c) = Likelihood Probability of event X occuring when event c has occured.
# P(c) = Probability of event c
# P(X) = Probability of event X

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Let's perform Naive Bayes on a Heart disease dataset.


# In[3]:


data=pd.read_excel(r'C:\Users\HP\Desktop\heartdata.xlsx')
data.head()


# In[4]:


# Shape of data
data.shape 


# In[5]:


# Information of the data
data.info()


# In[6]:


# Checking Null values
data.isnull().sum()


# In[7]:


# There are no null values present in the data.


# ## Data Analysis

# In[8]:


pd.crosstab(data.age,data.target).plot(kind='bar',figsize=(8,6),stacked=True)
plt.ylabel('No. of Observation')
plt.show()


# In[9]:


# We can see most of the heart patient lie in the 41 to 60 age group.


# In[10]:


data.groupby('target')['age'].mean().plot(kind ='pie', autopct='%f',explode=[.01,0.02])
plt.show()


# In[11]:


# we can conclude that the average age of heart disease patient's is 48 and the avg. of person's who have not 
# heart disese is 51%.


# In[12]:


data.groupby('sex')['target'].count().plot(kind='bar',color=['r','y'])
plt.ylabel('Count of people')
plt.show()


# In[13]:


# Here 0 represent female and 1 represent male. we can conclude that males are more heart patient than females.


# In[14]:


sns.distplot(data.trestbps)
plt.show()


# In[15]:


# We can see the blood pressure  between 120 to 140 are highly dominating among peoples.


# In[16]:


plt.scatter(data.age,data.trestbps)
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.show()


# In[17]:


# We can see as the age of people is increasing, the blood pressure is also increasing.


# In[18]:


plt.scatter(data.age, data.chol)
plt.xlabel('Age')
plt.ylabel('Cholestrol')
plt.show()


# In[19]:


# It also shows the linear relationship as the age is increasing, cholestrol is also incresing. We can also interpret the
# the person having highest cholestrol level is close to Age 70s'.


# In[20]:


# Let's see the correlation graph
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot = True)
plt.show()


# In[21]:


# Let's plot a Pairplot to seen the relationship among all variables.
sns.pairplot(data)
plt.show()


# ## Data Pre-Processing

# In[22]:


# Let's divide the data into numerical and categorical variables


# In[23]:


data.head()


# In[24]:


num_col = data[['age','trestbps','chol','thalach','oldpeak']]
cat_col = data[['sex','cp','fbs','restecg','exang','slope','ca','thal','target']]


# In[25]:


# Numerical data


# In[26]:


# Let's check for the outliers if there are any


# In[27]:


for i in num_col:
    sns.boxplot(num_col[i])
    plt.show()


# In[28]:


# We can check outliers by going through the outliers as shown Cholestrol and Blood Pressure features have some outliers. 
# As they are less in numbers, we can go with these outliers.


# In[29]:


# Categorical data


# In[30]:


cat_col.head()


# In[31]:


# We can see, all the cat feature are label encoded, if they were in nominal form,then we have to encoded the values 
# wheather we could use Label encoding or one-hot encoding.


# In[32]:


# One-Hot Encoding -

# Ex- A feature have three class 1, 2, 3. Feature values are (1,2,1,3,2,3,1).
# It assigns the weight like: Feature_1, Feature_2, Feature_3
#                                1         0          0                 As here feature value is 1
#                                0         1          0                 As here feature value is 2
# Similarily,                    1         0          0
#                                0         0          1
#                                0         1          0
#                                0         0          1
#                                1         0          0

# cat_col_dummies = pd.get_dummies(cat_col, drop_first=True)

# Get dummies is used for data manipulation and assign a indicator variable, similar like one-hot coding. 
# It split data feature into the no. of classes in that particular feature.
# Above, we use a parameter 'drop_first', It deletes the first dummy feature as we can get to know by looking other two
# indicatore if both are assigned 0 then the third one is assigned 1, similarily, if one is 0 and other is 1 the the 
# third assigned value is definately 0.


# In[33]:


# concatinating numerical and categorical data
all_data = pd.concat([num_col,cat_col],1)
all_data.head(2)


# In[34]:


X = data.drop('target',1)
y = data.target


# In[35]:


# Splitting data into test and train
from sklearn.model_selection import train_test_split


# In[36]:


xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=2)
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


# In[37]:


# importing Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,plot_roc_curve,roc_auc_score


# In[38]:


nb = GaussianNB()
nb.fit(xtrain,ytrain)


# In[39]:


ypred = nb.predict(xtrain)


# In[40]:


# Confusion Matrix
confusion_matrix(ytrain,ypred)


# In[41]:


# Classification Report
print(classification_report(ytrain,ypred))


# In[42]:


# Precision
pscore = precision_score(ytrain,ypred)
pscore


# In[43]:


# Recall
rscore = recall_score(ytrain,ypred)
rscore


# In[44]:


# accuracy
accuracy = accuracy_score(ytrain,ypred)
accuracy


# In[45]:


# ROC-AUC 
roc_score = roc_auc_score(ytrain,ypred)
roc_score


# In[46]:


# Plotting roc curve
plot_roc_curve(nb,xtrain,ytrain)
plt.show()


# In[47]:


# Adding above scores in dataframe
score = pd.DataFrame()
score['Train_Scores'] = [accuracy,pscore,rscore,roc_score]
score.index = ['Accuracy','Precision','Recall','ROC_AUC']
score


# In[48]:


# Let's test our model on test data


# In[49]:


# Accuracy
ypred_test = nb.predict(xtest)
test_accuracy = accuracy_score(ytest,ypred_test)
test_accuracy


# In[50]:


# Precision
test_pscore = precision_score(ytest,ypred_test)
test_pscore


# In[51]:


# Recall
test_rscore = recall_score(ytest,ypred_test)
test_rscore


# In[52]:


# ROC-AUC 
test_roc_score = roc_auc_score(ytest,ypred_test)
test_roc_score


# In[53]:


# plotting roc curve for test data
plot_roc_curve(nb,xtest,ytest)
plt.show()


# In[54]:


# Let's add these test scores in dataframe
score['Test_score'] = [test_accuracy,test_pscore,test_rscore,test_roc_score]
score


# In[55]:


# from the above results, we can see there are not much variance between train and test scores.


# ## Cross Validation
# 
# It is the method to validating the accuracy of the model by training it on different subsets of input data and testing on different unseen input data. We can not always rely on a single training data score, just beacause if our model face other  test data, it could be confused and could give the results that would hard to believe. So, to tackle this problem Cross Validation method came in use and able to provide us the robust result.
# 
# ### K-fold Cross Validation Method
# 
# It is a type of cross validation, here 'K' is the no. of fold that we input. Basically, what it does, It splits the dataset into K no. of folds without replacement. 
# 
# Steps taken by K-Fold:
# 
# Let assume K1, K2, K3...Kn are n no.of folds, It reserved K1 fold for test validation and others for Cross Validation training sets. After training the model on subsets, it calculates the accuracy on validation set.
# 
# Now, It takes k2 for test Validation and others for training sets. After training the model on all subsets, it calculates the accuracy on reserved validation set.
# 
# It follows the process until it trains all the subsets and evalutes the accuracy on each validation set and provides us robust accuracy score.

# In[56]:


from sklearn.model_selection import KFold, cross_val_score


# In[57]:


metric = ['accuracy','precision','recall','roc_auc']   #creating list of metrices, bcz CV takes a single metric at one go.
kf = KFold(n_splits=10,shuffle = True,random_state=2)  # creating K-Fold where n_splits is no. of folds.
cv = pd.DataFrame()                                    # Creating empty dataframe to record our results

for i in metric:
    cv_score = cross_val_score(nb,xtrain,ytrain,cv = kf,scoring = i)
    bias = np.mean(1-cv_score)
    var = np.std(cv_score,ddof = 1)
    cv[i] = [bias,var]

cv.index = ['Bias_Error','Var_Error']
cv


# In[58]:


# We can interpret the results as 'ROC_AUC' metric gave the better result than other metrics. It has low bias and low 
# variance among others.


# In[59]:


# We can also see for the test data
cv_test = pd.DataFrame()

for i in metric:
    cv_score = cross_val_score(nb,xtest,ytest,cv = kf,scoring = i)
    bias = np.mean(1-cv_score)
    var = np.std(cv_score,ddof = 1)
    cv_test[i] = [bias,var]

cv_test.index = ['Bias_Error','Var_Error']
cv_test


# In[60]:


# Here also 'ROC_AUC' gave the better result than others.

# We do not need to see results for each metric, we have to choose appropriate metric according to the problem statement or
# the dataset that we are working on. 
# We did this just for education purpose.


# ## K-Nearest Neighbour(KNN) Model
# 
# It is a type of Supervised Learning. It can be used for Classification as well as Regression problems. However, It is mainly used for classification problem. It is a non-parametric model which finds the similarity between test cases and the available cases and put the test cases into that category which is much similar to the available cases. It is a lazy lerning algorith because it does not learn at the training period instead it stores the dataset and starts learning when a test data is provided to classify the object.
# 
# **Steps:**
# 
# 1. First, we need to choose the optimal value of K that is nearest neighbours of a data point.
# 
# 2. Then, it calculates the distance between the test data and its neighbours using Manhatten distance or Euclidean distance. Most commonly, we use Euclidean method to calculate the distance.
# 
# 3. After calculating the distance, it finds the nearest K-neighbours to the test data points.
# 
# 4. It uses voting method to classify the object class meaning that if it finds 5 data points nearest to the test data, 3 of them belongs to class A and rest two points belongs to the class B then it will assign that test data point in Class A category. It assigns the category according to the majority or highest domination of a class nearest to the test data point.

# In[1]:


# Let's see graphical representation of KNN
# from IPython import display
# display.Image('D:\Python\k-nearest-neighbor.png')


# In[62]:


# imporing KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()  # by default k's value is 5
knn.fit(xtrain,ytrain)


# In[63]:


ypred = knn.predict(xtrain)


# In[64]:


print(classification_report(ytrain,ypred))


# In[65]:


# Accuracy
acc = accuracy_score(ytrain,ypred)
acc


# In[66]:


# Precision
pscore = precision_score(ytrain,ypred)
pscore


# In[67]:


#Recall
rscore = recall_score(ytrain,ypred)
rscore


# In[68]:


# ROC-AUC Score
roc_score = roc_auc_score(ytrain,ypred)
roc_score


# In[69]:


plot_roc_curve(knn,xtrain,ytrain)


# In[70]:


knn_scores = pd.DataFrame()
knn_scores['Train Score'] = [acc,pscore,rscore,roc_score]
knn_scores.index= ['Accuracy','Precision','Recall','ROC_AUC']
knn_scores


# In[71]:


# Testing model
ypred_t = knn.predict(xtest)
acc_t = accuracy_score(ytest,ypred_t)
acc_t


# In[72]:


# Precision
pscore_t = precision_score(ytest,ypred_t)
pscore_t


# In[73]:


# Recall
rscore_t = recall_score(ytest,ypred_t)
rscore_t


# In[74]:


# ROC-AUC
roc_t = roc_auc_score(ytest,ypred_t)
roc_t


# In[75]:


knn_scores['Test Score'] = [acc_t,pscore_t,rscore_t,roc_t]
knn_scores


# In[76]:


# We can see the variance is too high between traning and testing results.
# Let's use cross validation method and see results


# In[77]:


# Cross Validation for train data
metric = ['accuracy','precision','recall','roc_auc'] 
kf = KFold(n_splits=10,shuffle = True,random_state=2)  
knn_cv = pd.DataFrame()

for i in metric:
    cv_score = cross_val_score(nb,xtrain,ytrain,cv = kf,scoring = i)
    bias = np.mean(1-cv_score)
    var = np.std(cv_score,ddof = 1)
    knn_cv[i] = [bias,var]

knn_cv.index = ['Bias_Error','Var_Error']
knn_cv


# In[78]:


# Cross Validation for test data
metric = ['accuracy','precision','recall','roc_auc'] 
kf = KFold(n_splits=10,shuffle = True,random_state=2)  
knn_cv_test = pd.DataFrame()

for i in metric:
    cv_score = cross_val_score(nb,xtest,ytest,cv = kf,scoring = i)
    bias = np.mean(1-cv_score)
    var = np.std(cv_score,ddof = 1)
    knn_cv_test[i] = [bias,var]

knn_cv_test.index = ['Bias_Error','Var_Error']
knn_cv_test


# In[79]:


# We can see the CV results, the variance between the train and test results become less.


# In[80]:


# We can also choose the value of K using GridSearch library.


# ## GridSearchCV
# 
# It is python function which helps in finding the right parameter value. Here we use it to find the value of K.

# In[81]:


from sklearn.model_selection import GridSearchCV


# In[82]:


# First we need to make a dictionary having the parameters which we need to find.
params = {'n_neighbors':np.arange(3,100),'weights':['uniform','distance']}
# we set n_neighbors from 3 to 50 because it should be min. 3 for better result.
# knn use two kind of weights:
# Uniform - It assigns the weight equally all nearest neighbors.
# Distance - It assigns the more weight to the most near data point and less to the farthest data point.
GS = GridSearchCV(knn,params,cv = 5,scoring = 'roc_auc')
GS.fit(xtrain,ytrain)   


# In[83]:


GS.best_params_


# In[84]:


# Here we find the value of K = 87 and weight is distance.


# In[85]:


knn_tunned = KNeighborsClassifier(n_neighbors=87,weights='distance') 
# This process is so called tunning if we set parameter to get the better results.
knn_tunned.fit(xtrain,ytrain)


# In[86]:


ypred = knn_tunned.predict(xtrain)


# In[87]:


roc_auc_score(ytrain,ypred)


# In[88]:


print(classification_report(ytrain,ypred))

