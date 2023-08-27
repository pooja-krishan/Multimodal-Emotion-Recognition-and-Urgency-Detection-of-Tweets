#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np
import seaborn as sns
import nltk as nl
import spacy
import re
from textblob import Word
nl.download('punkt')
nl.download('averaged_perceptron_tagger')
#nlp = spacy.load("en_core_web_sm")
nl.download('stopwords')


# In[4]:


text_df = pd.read_csv('IEMOCAP_TEXT/text_features_extraction.csv')


# In[5]:


text_df.head()


# In[6]:


text_df = text_df.drop(["filename","text"],axis=1)
text_df = text_df.drop("Unnamed: 0",axis=1)


# In[7]:


text_df = text_df.drop("processed_text",axis=1)
text_df.head()


# In[12]:


text_df = text_df[text_df.emotion != "xxx"]
text_df = text_df[text_df.emotion != "fru"]
text_df = text_df[text_df.emotion != "sur"]
text_df = text_df[text_df.emotion != "exc"]
text_df = text_df[text_df.emotion != "oth"]
text_df = text_df[text_df.emotion != "fea"]
text_df = text_df[text_df.emotion != "dis"]
text_df.head()


# In[13]:


import seaborn as sns
sns.countplot(x="emotion", data=text_df)
plt.show()


# In[14]:


from sklearn.model_selection import train_test_split
X = text_df.drop(['emotion'],axis=1).values
y = text_df['emotion'].values


# In[15]:


X.shape


# In[16]:


y.shape


# In[ ]:





# In[17]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[18]:


rf = RandomForestClassifier(max_depth=2, random_state=42)
rf.fit(X, y)
rf.feature_importances_  
model = SelectFromModel(rf, prefit=True)
X = model.transform(X)
X.shape


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=6, stratify=y)
print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(y_test.shape)


# In[20]:


Counter(y_train)


# In[58]:


#from imblearn.under_sampling import RandomUnderSampler
# define undersampling strategy
#undersample = RandomUnderSampler(sampling_strategy='majority')

# fit and apply the transform
#X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

# summarize class distribution
#print("After undersampling: ", Counter(y_train_under))


# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[22]:


# Create the models to be tested
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)

rf = RandomForestClassifier(max_depth=2, random_state=42)
rf.fit(X_train, y_train)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
gbc.fit(X_train, y_train)

# Put the models in a list to be used for Cross-Validation
models = [logreg, rf, gbc]

from sklearn.model_selection import cross_val_score

def evaluate_model(model):

        
        accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        print(np.mean(accuracy))


# In[23]:


#Undersampling set
"""
logreg_under = LogisticRegression(max_iter=5000)
logreg_under.fit(X_train_under, y_train_under)

rf_under = RandomForestClassifier(max_depth=2, random_state=42)
rf_under.fit(X_train_under, y_train_under)

gbc_under = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
gbc_under.fit(X_train_under, y_train_under)

 Put the models in a list to be used for Cross-Validation
models_under = [logreg_under, rf_under, gbc_under]

evaluate_model(logreg_under)
evaluate_model(rf_under)
evaluate_model(gbc_under)
"""


# In[24]:


evaluate_model(logreg)
evaluate_model(rf)
evaluate_model(gbc)


# In[25]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# In[26]:


model = LogisticRegression(max_iter = 5000)
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[27]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[28]:


# define models and parameters
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[29]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


# In[30]:


# define models and parameters
model = GradientBoostingClassifier()
n_estimators = [10, 100]
learning_rate = [0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[31]:


# Create the final Multiple Linear Regression
logreg_final = LogisticRegression(C=1.0, penalty = 'l2', solver= 'liblinear')
# Create the final Random Forest
rf_final = RandomForestClassifier(n_estimators = 1000,   
                                 max_features = 'log2',
                                 random_state = 42)

# Create the fnal Extreme Gradient Booster
gbc_final = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42, subsample =1.0)

# Train the models using 80% of the original data
logreg_final.fit(X_train, y_train)
rf_final.fit(X_train, y_train)
gbc_final.fit(X_train, y_train)


# In[32]:


evaluate_model(logreg_final)
evaluate_model(rf_final)
evaluate_model(gbc_final)


# In[33]:


from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix,classification_report


# In[34]:


# LogReg.score(X,y)
model_train = logreg_final.predict(X_train)
model_test = logreg_final.predict(X_test)
print(logreg_final.score(X_train,y_train))
print(logreg_final.score(X_test,y_test))


# In[35]:


color = 'white'
matrix = plot_confusion_matrix(logreg_final, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.show()
confusion_matrix = metrics.confusion_matrix(y_test, model_test)
confusion_matrix


# In[36]:


report = classification_report(y_test, model_test,output_dict=True)
df_classification_report = pd.DataFrame(report).transpose()
df_classification_report


# In[37]:


from sklearn.metrics import plot_confusion_matrix
 
# performing predictions on the test dataset
y_pred = rf_final.predict(X_test)
print(rf_final.score(X_train,y_train))
print(rf_final.score(X_test,y_test))
color = 'white'
matrix = plot_confusion_matrix(rf_final, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.show()
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix


# In[38]:


report = classification_report(y_test, y_pred,output_dict=True)
df_classification_report = pd.DataFrame(report).transpose()
df_classification_report


# In[39]:


from sklearn.metrics import plot_confusion_matrix
y_predict = gbc_final.predict(X_test)
print(gbc_final.score(X_train,y_train))
print(gbc_final.score(X_test,y_test))
color = 'white'
matrix = plot_confusion_matrix(gbc_final, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.show()
confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
confusion_matrix


# In[40]:


report = classification_report(y_test, y_predict,output_dict=True)
df_classification_report = pd.DataFrame(report).transpose()
df_classification_report


# In[ ]:




