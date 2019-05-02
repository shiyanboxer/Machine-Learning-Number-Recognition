#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import modules
import matplotlib.pyplot as plt #plots data 
from sklearn import datasets #
from sklearn import svm #Support Vector Machine.- categorize data sets


# In[2]:


#define digits variables - loaded digit dataset.
digits = datasets.load_digits()
#the actual data (features)
print(digits.data)


# In[3]:


# the actual label we've assigned to the digits data
print(digits.target)


# In[7]:


#specify the classifier parameters - this chooses the SVC, and we set gamma and C
clf = svm.SVC(gamma=0.001, C=100)


# In[22]:


# loads in all but the last 10 data points, so we can use all of these for training. Then, we can use the last 10 data points for testing
X,y = digits.data[:-1], digits.target[:-1]
#train the machine
clf.fit(X,y)


# In[25]:


#print results
print("Prediction:",clf.predict(digits.data[-1]))


# In[24]:


#display image
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# In[ ]:




