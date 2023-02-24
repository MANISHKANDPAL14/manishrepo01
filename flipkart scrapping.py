#!/usr/bin/env python
# coding: utf-8

# # Flipkart Data scrapping of mobile phones under 50,000

# In[1]:


#importing necessary libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup


# In[2]:


##url of the flipkart page
url="https://www.flipkart.com/search?q=mobile%20phone%20under%2050000&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"


# In[3]:


# getting the url 
r=requests.get(url)
r


# In[4]:


soup=BeautifulSoup(r.text,"lxml")
soup.head()
# large irregular data appears here


# In[5]:


np=soup.find("a",class_="_1LKTO3").get("href")
np


# In[6]:


cnp="https://www.flipkart.com"+np
print(cnp)


# In[7]:


url=cnp
r=requests.get(url)
soup=BeautifulSoup(r.text,"lxml")
soup
#irregular data appears but somewhat better than before


# In[8]:


while True:
    np=soup.find("a",class_="_1LKTO3").get("href")
    cnp="https://www.flipkart.com"+np
    url=cnp 
    print(url)
    r=requests.get(url)
    soup=BeautifulSoup(r.text,"lxml")
# here page 1 and 2 are repeated continiously instead of all pages display


# In[ ]:


for i in range(2,12):
    url="https://www.flipkart.com/search?q=mobile+phone+under+50000&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&page="+str(i)
    print(url)
    r=requests.get(url)
    soup=BeautifulSoup(r.text,"lxml")
    soup


# In[ ]:


product_name=[]
price=[]
description=[]
review=[]
for i in range(2,11):
    url="https://www.flipkart.com/search?q=mobile+phone+under+50000&otracker=FLIPKART&as-show=on&as=off&page="+str(i)
    print(url)
    r=requests.get(url)
    soup=BeautifulSoup(r.text,"lxml")
    box=soup.find("div",class_="_1YokD2 _3Mn1Gg")
    names=box.find_all("div",class_="_4rR01T")
    prices=box.find_all("div",class_="_30jeq3 _1_WHN1")
    descriptions=box.find_all("ul",class_="_1xgFaf")
    reviews=box.find_all("div",class_="_3LWZlK")
    for i in names:
        name=i.text
        product_name.append(name)
    print(product_name)

    for i in prices:
        name=i.text
        price.append(name)
    print(price)

    for i in descriptions:
        name=i.text
        description.append(name)
    print(description) 

    for i in reviews:
        name=i.text
        review.append(name)
    print(review)


# In[ ]:


print(len(list(product_name)))
print(len(list(price)))
print(len(list(review)))
print(len(list(description)))


# In[ ]:


#creating the dataframe of all data so that it looks beautiful and organised in columns
df1=pd.DataFrame({"product names":product_name,"prices":price,"descriptions":description,"reviews":review})
df1


# In[ ]:


#now converting the dataframe to csv file and then saving it at location F drive under scrapping folder and file name fbscrapping
data1=df1.to_csv("F:/scrapping/fbscrapping2.csv")
data1


# In[ ]:


#importing necessary libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
product_name=[]
price=[]
description=[]
review=[]
for i in range(2,11):
    url="https://www.flipkart.com/search?q=mobile+phone+under+50000&otracker=FLIPKART&as-show=on&as=off&page="+str(i)
    soup=BeautifulSoup(requests.get(url).text,"lxml")
    box=soup.find("div",class_="_1YokD2 _3Mn1Gg")
    names=box.find_all("div",class_="_4rR01T")
    prices=box.find_all("div",class_="_30jeq3 _1_WHN1")
    descriptions=box.find_all("ul",class_="_1xgFaf")
    reviews=box.find_all("div",class_="_3LWZlK")
    for i in names:
        product_name.append(i.text)
    #print(product_name)

    for i in prices:
          price.append(i.text)
    #print(price)

    for i in descriptions:
        description.append(i.text)
    #print(description) 

    for i in reviews:
        review.append(i.text)
    #print(review)
    
print(len(list(product_name)))
print(len(list(price)))
print(len(list(review)))
print(len(list(description)))

dff=pd.DataFrame({"product names":product_name,"prices":price,"descriptions":description,"reviews":review})
dff    

