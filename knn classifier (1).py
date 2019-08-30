#!/usr/bin/env python
# coding: utf-8

# In[1]:


X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


neigh = KNeighborsClassifier(n_neighbors=3)


# In[3]:


neigh.fit(X, y)


# In[7]:


print(neigh.predict([[3]]))


# In[8]:


print(neigh.predict_proba([[0.9]]))


# # Classifier Building in Scikit-learn
# 

# In[10]:


# Assigning features and label variables
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Label or target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# # Encoding data columns

# In[11]:


# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print(weather_encoded)


# In[12]:


# converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)


# In[17]:


print(temp_encoded)
print(label)


# # Combining Features
# 

# In[15]:


#combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded))


# # Generating Model

# In[18]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print(predicted)


# # KNN with Multiple Labels

# In[31]:


#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
wine = datasets.load_wine()


# In[32]:


# print the names of the features
print(wine.feature_names)


# In[33]:


# print the label species(class_0, class_1, class_2)
print(wine.target_names)


# In[34]:


# print the wine data (top 5 records)
print(wine.data[0:5])


# In[35]:


# print the wine labels (0:Class_0, 1:Class_1, 2:Class_3)
print(wine.target)


# In[36]:


print(wine.data.shape)


# In[37]:


# print target(or label)shape
print(wine.target.shape)


# # Splitting Data

# In[38]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test


# # Generating Model for K=5
# Let's build KNN classifier model for k=5.

# In[54]:


#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=9)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


# In[55]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # Model Evaluation for k=7

# In[56]:


#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


# In[57]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




