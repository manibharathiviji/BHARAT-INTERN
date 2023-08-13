#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


train_pd = pd.read_csv("C:/Users/ambri/OneDrive/Desktop/python/train.csv")
train_pd


# In[5]:


train_pd = train_pd[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
train_pd


# In[6]:


train_pd = train_pd.dropna()
train_pd


# In[7]:


train_pd.describe()


# In[8]:


from tensorflow.keras.utils import to_categorical
pclass = train_pd['Pclass'].to_numpy()
as_categorical_pclass = to_categorical(pclass - 1, num_classes=3)
as_categorical_pclass


# In[9]:


def sex_str_to_binary(s):
    if s=="male":
        return 1
    else:
        return 0
sex = train_pd['Sex']

as_binary_sex = sex.apply(sex_str_to_binary).to_numpy()
as_binary_sex[:10]


# In[10]:


age_np = train_pd['Age'].to_numpy()
age_np = (age_np - train_pd['Age'].mean()) / train_pd['Age'].std()
age_np[:10]


# In[11]:


sibsp = train_pd['SibSp']
sibsp_np = sibsp.to_numpy()
sibsp_np = (sibsp_np - train_pd['SibSp'].mean()) / train_pd['SibSp'].std()
sibsp_np[:10]


# In[12]:


parch = train_pd['Parch']
parch = parch.to_numpy()
parch = (parch - train_pd['Parch'].mean()) / train_pd['Parch'].std()
parch[:10]


# In[13]:


fare = train_pd['Fare'].to_numpy()
fare = (fare - train_pd['Fare'].mean()) / train_pd['Fare'].std()
fare[:10]


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


train_Y = train_pd['Survived'].to_numpy()
train_Y[:10]


# In[16]:


train_X = np.concatenate((as_categorical_pclass, 
                as_binary_sex.reshape(-1, 1),
                age_np.reshape(-1, 1),
                sibsp_np.reshape(-1,1),
                parch.reshape(-1,1),
                fare.reshape(-1,1)), axis=1)
train_X


# In[17]:


model = LogisticRegression(random_state=0).fit(train_X, train_Y)
model.score(train_X, train_Y)


# In[18]:


test_pd = pd.read_csv("C:/Users/ambri/OneDrive/Desktop/python/test.csv")
test_pd


# In[19]:


test_pd = test_pd[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
test_pd


# In[20]:


test_pd = test_pd.fillna(test_pd['Age'].mean())
len(test_pd)


# In[21]:


pclass = test_pd['Pclass'].to_numpy()
as_categorical_pclass = to_categorical(pclass - 1, num_classes=3)
as_categorical_pclass[:10]


# In[22]:


sex = test_pd['Sex']
as_binary_sex = sex.apply(sex_str_to_binary).to_numpy()
as_binary_sex[:10]


# In[23]:


age_np = test_pd['Age'].to_numpy()
age_np = (age_np - train_pd['Age'].mean()) / train_pd['Age'].std()
age_np[:10]


# In[24]:


sibsp = test_pd['SibSp']
sibsp_np = sibsp.to_numpy()
sibsp_np = (sibsp_np - train_pd['SibSp'].mean()) / train_pd['SibSp'].std()
sibsp_np[:10]


# In[25]:


parch = test_pd['Parch']
parch = parch.to_numpy()
parch = (parch - train_pd['Parch'].mean()) / train_pd['Parch'].std()
parch[:10]


# In[26]:


fare = test_pd['Fare'].to_numpy()
fare = (fare - train_pd['Fare'].mean()) / train_pd['Fare'].std()
fare[:10]


# In[27]:


test_X = np.concatenate((as_categorical_pclass, 
                as_binary_sex.reshape(-1, 1),
                age_np.reshape(-1, 1),
                sibsp_np.reshape(-1,1),
                parch.reshape(-1,1),
                fare.reshape(-1,1)), axis=1)
test_X


# In[31]:


predictions = model.predict(test_X)
passenger_ids = test_pd['PassengerId'].to_numpy()
sub_df = pd.DataFrame({'PassengerId':passenger_ids, 'Survived':predictions})


# In[33]:


sub_df


# # ---------AMBRISH------------
