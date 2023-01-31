#!/usr/bin/env python
# coding: utf-8

# We will use the mushroom dataset from UCI Machine Learning Repository and examine whether mushrooms are edible or poisonous. The dataset has the following attributes:
# 1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises?: bruises=t,no=f
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 6. gill-attachment: attached=a,descending=d,free=f,notched=n
# 7. gill-spacing: close=c,crowded=w,distant=d
# 8. gill-size: broad=b,narrow=n
# 9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
# 10. stalk-shape: enlarging=e,tapering=t
# 11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 16. veil-type: partial=p,universal=u
# 17. veil-color: brown=n,orange=o,white=w,yellow=y
# 18. ring-number: none=n,one=o,two=t
# 19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
# 21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
# 22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv("mushrooms.csv")
df.head()


# In[14]:


df.info()


# #### Converting categorical data using LabelEncoder

# In[16]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


# In[17]:


df.head()


# In[42]:


df.describe()


# #### EDA

# In[11]:


sns.countplot(x=df['class'])


# We can see that there are more edible mushrooms than poisnous ones in our dataset.

# In[43]:


plt.figure(figsize=(16,16))
sns.heatmap(df.corr(),annot=True)


# Dropping viel-type column as it doesn't provide any information

# In[20]:


df = df.drop(["veil-type"],axis=1)


# In[23]:


df.groupby(['habitat', 'class']).size().unstack().plot.bar(stacked=True) 
plt.show()


# We can see that the most poisonous mushrooms are found in grasses(0). Also, mushrooms found in the woods (6) are unlikely to harmful. 

# In[24]:


df.groupby(['population', 'class']).size().unstack().plot.bar(stacked=True)


# Also, mushrooms that are scattered are likely to be poisounous compared to those found in solitary or group. 

# #### Train, Test split

# In[25]:


X = df.drop(['class'], axis=1)
y = df['class']


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[27]:


from sklearn.svm import SVC


# In[28]:


svm = SVC()
svm.fit(X_train, y_train)


# #### Evaluation

# In[29]:


pred = svm.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report


# In[31]:


print(classification_report(y_test,pred))


# #### Cross Validation

# In[32]:


from sklearn.model_selection import cross_val_score


# In[34]:


accuracies = cross_val_score(svm,X,y, cv = 10)


# In[39]:


print ('Accuracies',accuracies)
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))


# In[1]:



# Hence, our model doesn't overfit and works well on new data.
