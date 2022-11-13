
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston 
boston = load_boston()


# In[2]:



print(boston)


# In[3]:


print(boston.data.shape)


# In[4]:


print(boston.feature_names)
print(boston.target)


# In[5]:



print(boston.DESCR)


# In[6]:


import pandas as pd
bos=pd.DataFrame(boston.data)
print(bos.head())



# In[7]:


bos['PRICE']=boston.target
X=bos.drop('PRICE',axis=1)
Y=bos['PRICE']
print(Y)


# In[8]:


print(X)


# In[10]:


import sklearn 
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[14]:


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
lm=LinearRegression()
lm.fit(X_train,Y_train)
Y_Pred=lm.predict(X_test)
plt.scatter(Y_test,Y_Pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices:$Y_i$ vs $\hat{Y}_i$")
plt.show()


# In[12]:


delta_y = Y_test - Y_Pred;
import seaborn as sns;
import numpy as np;
sns.set_style('whitegrid')
sns.kdeplot(np.array(delta_y), bw=0.8)
plt.show()


# In[13]:


sns.set_style("whitegrid")
sns.kdeplot(np.array(Y_Pred),bw=0.5)
plt.show()

