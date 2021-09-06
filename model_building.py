# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 06:32:00 2021

@author: kapil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("eda_data.csv")

#Choose relevant columns

df.columns
df_model = df[["Avg_Salary","Rating", "Size", "Type of ownership", "Industry", "Sector",
             "Revenue", 'Num_Comp','Hourly','Employer Provided','Job_State', 
             'Same_State', 'Age', 'Python_yn', 'Spark_yn', 'Aws_yn', 'Excel_yn',
             'Job_Simp','Seniority','Desc_Len',]]


#Get dummy data

df_dum = pd.get_dummies(df_model)


#Train test split

from sklearn.model_selection import train_test_split
X = df_dum.drop('Avg_Salary', axis = 1)
y = df_dum.Avg_Salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)


#Multiple linear regression

import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring = "neg_mean_absolute_error", cv=3))


#Lasso regression

lm_l = Lasso(alpha = 0.103)  # Assigned this value of alpha after finding best score
lm_l.fit(X_train, y_train)

np.mean(cross_val_score(lm_l, X_train, y_train, scoring = "neg_mean_absolute_error", cv = 3))

alpha = []
error = []

for i in range(1,1000):
    alpha.append(i/1000)
    lml = Lasso(alpha = (i/1000))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring = "neg_mean_absolute_error", cv=3)))

plt.plot(alpha,error)  # The peak of the graph shows that at alpha < 0.2 gives best error value,

#We will find the exact value next:
    
err = tuple(zip(alpha, error)) 
df_err = pd.DataFrame(err, columns=['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

#This shows that we can tune our model by finding accurate value of alpha = 0.103
#Need to put this alpha = 0.103 in Lasso to get the best score 

  
#Random forest

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring ="neg_mean_absolute_error", cv=3))
#Better than previous model, but still needs tuning : -15.243437761711874

#Tune models using GridsearchCV

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv = 3)
gs.fit(X_train, y_train)

gs.best_score_    #Score improved a little -14.933199806867322
gs.best_estimator_  #Gives us the parameters that generated the best score : RandomForestRegressor(criterion='mae', n_estimators=140)


#Test ensembles i.e. we will check with the above 3 models by predicting the results on Test dataset and see if we get the same results
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)    #19.366239435977473
mean_absolute_error(y_test, tpred_lml)   #20.186588088262386
mean_absolute_error(y_test, tpred_rf)    #11.498633748801534   Performed best

#We can try to tune it further by combining two best performing models and see if it helps:
mean_absolute_error(y_test, (tpred_lm+tpred_rf)/2)    #14.736516615746968    not better than RandomForest










