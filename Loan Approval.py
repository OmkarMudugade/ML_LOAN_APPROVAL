#!/usr/bin/env python
# coding: utf-8

# In[162]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
 
import warnings
warnings.filterwarnings('ignore')


# In[163]:


df = pd.read_excel(r"C:\Users\dell\Downloads\Copy of loan1.xlsx")


# In[164]:


df


# In[165]:


df.info()


# In[166]:


df.describe().T


# In[167]:


df.columns


# In[168]:


df.isnull().sum()


# In[169]:


df['loanAmount_log'] = np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20)


# In[170]:


df['TotalIncome']= df['ApplicantIncome']+ df['CoapplicantIncome']

df['TotalIncome_log']= np.log(df['TotalIncome'])

df['TotalIncome_log'].hist(bins=20)


# In[171]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
df['Married'].fillna(df['Married'].mode()[0], inplace = True)

df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace = True)

df['Dependents'].fillna(df['Dependents'].mode()[0], inplace = True)


# In[172]:


df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())


# In[173]:


df.loanAmount_log = df.loanAmount_log.fillna(df.loanAmount_log.mean())


# In[174]:


df['Loan_Amount_Term'].fillna(df[ 'Loan_Amount_Term'].mode()[0], inplace = True) 
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace = True)


# In[175]:


df.isnull().sum()


# In[176]:


df


# In[ ]:





# In[177]:


categorical = ['Gender','Married','Education','Property_Area','Self_Employed']
df = pd.get_dummies(df, columns=categorical, drop_first=False)
df


# In[178]:


column_to_move = 'Loan_Status'

# Move the selected column to the last position
df = df[[col for col in df.columns if col != column_to_move] + [column_to_move]]

# Display the DataFrame after moving the column
df


# In[179]:


X=df.iloc[:,1:21]


# In[180]:


X


# In[181]:


Y=df.iloc[:,21:]


# In[182]:


Y


# In[183]:


#spliting the data into train and test
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)


# In[184]:


X_train.shape 


# In[185]:


y_train.shape


# In[186]:


X_test.shape


# In[187]:


y_test.shape


# # KNN

# In[188]:


#applying knn

# knn(k nearest neighbour)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Create KNN classifier

hyper_parameter_k=[{'n_neighbors':[3,5,7,9,11,13]}]
clf= KNeighborsClassifier()
modelknn=GridSearchCV( clf,hyper_parameter_k,scoring='accuracy')

modelknn.fit(X_train,y_train)

print(modelknn.best_estimator_)
print("training accuracy is",modelknn.score(X_train,y_train))
print("testing accuracy is ",modelknn.score(X_test,y_test))


# # NAIVE BAYES

# In[189]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)


# In[190]:


print("Naive Bayes score: ",nb.score(X_test, y_test))


# # DT

# In[197]:


from sklearn import tree
tuned_parameters=[{'max_depth':[4,5,6]}]

clf = tree.DecisionTreeClassifier(min_samples_split=5,random_state=42)


dtmodel=GridSearchCV(clf,tuned_parameters)

dtmodel.fit(X_train,y_train)

print(dtmodel.best_estimator_)
print(dtmodel.score(X_test,y_test))
print(dtmodel.score(X_train,y_train))


# In[198]:


#PREDICTIONS
dt_test_prediction=dtmodel.predict(X_test)#this is your y_pred_test
dt_train_predictions=dtmodel.predict(X_train)#y_pred_train


# In[199]:


#Precison
#testing data
from sklearn.metrics import precision_score
x_dt_p_test=precision_score(y_test,dt_test_prediction,average='weighted')
print("The precision of x_dt_p for testing data",x_dt_p_test)


# In[200]:


#RECALL
#Test DATA
from sklearn.metrics import recall_score
x_dt_r_test=recall_score(y_test,dt_test_prediction,average='weighted')
print("The recall of test data for Decison Tree is",x_dt_r_test)


# In[201]:


#accuracy
from sklearn.metrics import accuracy_score
a=accuracy_score(y_train,dt_train_predictions)

print("training accuracy is",a)

from sklearn.metrics import accuracy_score
b=accuracy_score(y_test,dt_test_prediction)

print("testing accuracy is",b)


# In[ ]:





# # random forest

# In[191]:


#Training random forest after applying SMOTE technique.
#here we are using GridSearchCV to tune the hyperparamters we have in randomforest to see which 
# parameters works best.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [200,300, 500],
    'max_depth' : [4,5,6]
}
rfc=RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, 
                          cv = 3, n_jobs = -1)


# In[192]:


grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


# In[193]:


#training the random forest model
rfc_new=RandomForestClassifier(criterion= 'entropy', max_depth= 6, max_features= 'auto', n_estimators=500,random_state=42)
random_forest_model=rfc_new.fit(X_train,y_train)


# In[194]:


#predictions
rf_predictions_test=random_forest_model.predict(X_test)
rf_predictions_train=random_forest_model.predict(X_train)


# In[195]:


#evaluation of random forest with smote
from sklearn.metrics import f1_score
X_test_F1score=f1_score(y_test,rf_predictions_test,average='weighted')
print("Testing f1 score is",X_test_F1score) #f1 score on test data

X_train_F1score=f1_score(y_train,rf_predictions_train,average='weighted')
print("Training f1 score is",X_train_F1score) #f1 score on train data


# In[202]:


#accuracy
from sklearn.metrics import accuracy_score
a=accuracy_score(y_train,dt_train_predictions)

print("training accuracy is",a)

from sklearn.metrics import accuracy_score
b=accuracy_score(y_test,dt_test_prediction)

print("testing accuracy is",b)


# # LOGISTIC 

# In[203]:


#we are using simple gridsearchcv to tune the hyperparameter C
#here we are not using smote samples

tuned_parameters=[{'C':[10**-4,10**-2,10**0,10**2,10**4]}]

LRmodel=GridSearchCV(LogisticRegression(max_iter=400,class_weight='balanced'),tuned_parameters)

LRmodel.fit(X_train,y_train)

print(LRmodel.best_estimator_)
print(LRmodel.score(X_test,y_test))
print(LRmodel.score(X_train,y_train))

#this is accuracy 


# In[204]:


#PREDICTIONS
LR_test_prediction=LRmodel.predict(X_test)#this is your y_pred_test
LR_train_predictions=LRmodel.predict(X_train)#y_pred_train


# In[205]:


#evaluation
from sklearn.metrics import f1_score
X_LRtest_F1score=f1_score(y_test,LR_test_prediction,average='weighted')
print("Testing f1 score is",X_LRtest_F1score) #f1 score on test data

X_LRtrain_F1score=f1_score(y_train,LR_train_predictions,average='weighted')
print("Training f1 score is",X_LRtrain_F1score) #f1 score on train data


# In[206]:


#Precison
#testing data
from sklearn.metrics import precision_score
x_LR_p_test=precision_score(y_test,LR_test_prediction,average='weighted')
print("The precision of x_LR_p for testing data",x_LR_p_test)


# In[207]:


#RECALL
#Test DATA
from sklearn.metrics import recall_score
x_LR_r_test=recall_score(y_test,LR_test_prediction,average='weighted')
print("The recall of test data for Logistic Regression is",x_LR_r_test)


# In[208]:


#accuracy
from sklearn.metrics import accuracy_score
a=accuracy_score(y_train,LR_train_predictions)

print("training accuracy is",a)

from sklearn.metrics import accuracy_score
b=accuracy_score(y_test,LR_test_prediction)

print("testing accuracy is",b)


# In[ ]:




